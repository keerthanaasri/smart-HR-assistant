import streamlit as st
import re
import os
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart HR Assistant",
    page_icon="https://img.icons8.com/fluency/48/artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
        /* Main Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hides the default Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Custom Header */
        .main-header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.5rem;
        }
        .main-header img {
            width: 50px;
            margin-right: 15px;
        }
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4F46E5; /* Indigo 600 */
            margin: 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6B7280; /* Gray 500 */
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Feature Headers */
        .feature-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .feature-header img {
            width: 30px;
            margin-right: 10px;
        }
        .feature-header h3 {
            margin: 0;
        }

        /* Card Styling */
        .card {
            background-color: #1F2937; /* Gray 800 */
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1rem;
            border: 1px solid #374151;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111827; /* Gray 900 */
            border-right: 1px solid #374151;
        }

        /* Highlights */
        .highlight {
            color: #818CF8; /* Indigo 400 */
            font-weight: 600;
        }
        
    </style>
""", unsafe_allow_html=True)

# ---------------- CACHING THE MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- HELPER FUNCTIONS ----------------

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
        text = re.sub(r"\s+", " ", text)
        return text.lower()
    except Exception as e:
        return ""

def semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100

def extract_skills_from_jd(job_desc):
    words = re.findall(r"\b[A-Za-z\+\#\.]+\b", job_desc)
    stopwords = {"with","and","for","the","looking","experience","engineer","developer","required", "knowledge", "proficiency", "familiarity", "expertise", "understanding", "ability", "skills", "strong", "excellent", "good", "proven", "track", "record"}
    skills = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
    return list(set(skills))

def skill_transparency(resume_text, required_skills):
    matched = []
    for skill in required_skills:
        if re.search(rf"\b{re.escape(skill)}\b", resume_text):
            matched.append(skill)
    return matched

def ocean_personality_analysis(text):
    personality = {
        "Openness": 0,
        "Conscientiousness": 0,
        "Extraversion": 0,
        "Agreeableness": 0,
        "Emotional Stability": 0
    }
    if re.search(r"creative|innovation|research|curious|explore", text):
        personality["Openness"] = 20
    if re.search(r"organized|planning|deadline|responsible|detail|managed|delivered", text):
        personality["Conscientiousness"] = 20
    if re.search(r"lead|team|communicat|present|collaborat", text):
        personality["Extraversion"] = 20
    if re.search(r"help|support|cooperate|volunteer|friendly", text):
        personality["Agreeableness"] = 20
    if re.search(r"handled pressure|stress management|resilient|calm under pressure|crisis", text):
        personality["Emotional Stability"] = 20
    return personality

def extract_desired_traits(culture_text):
    traits = []
    if re.search(r"creative|innovation|research", culture_text):
        traits.append("Openness")
    if re.search(r"organized|discipline|planning|structured", culture_text):
        traits.append("Conscientiousness")
    if re.search(r"communication|leader|teamwork|collaboration", culture_text):
        traits.append("Extraversion")
    if re.search(r"helpful|cooperative|supportive|friendly", culture_text):
        traits.append("Agreeableness")
    if re.search(r"calm|stable|stress tolerance|resilient", culture_text):
        traits.append("Emotional Stability")
    return traits

def calculate_culture_fit(personality, desired_traits):
    return sum(personality.get(trait, 0) for trait in desired_traits)

def experience_level(text):
    # Numeric regex
    numeric_match = re.findall(r"(\d+)\+?\s*(years|yrs)", text)
    for match in numeric_match:
        years = int(match[0])
        if years >= 5: return "High"
        elif years >= 3: return "Medium"
    
    # Written numbers
    written_years = {"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for word,value in written_years.items():
        if re.search(rf"{word}\s+years", text):
            if value >= 5: return "High"
            elif value >= 3: return "Medium"
            
    # Date ranges
    date_ranges = re.findall(r"(20\d{2})\s*[-–]\s*(20\d{2}|present|current|now)", text)
    for start,end in date_ranges:
        start = int(start)
        end = datetime.now().year if end in ["present", "current", "now"] else int(end)
        diff = end - start
        if diff >= 5: return "High"
        elif diff >= 3: return "Medium"
    
    return "Low"

def project_match(text):
    return "High" if re.search(r"project|developed|built|implemented", text) else "Low"

def detect_skill_stuffing(text, required_skills):
    skill_mentions = 0
    # Simple count of total occurrences of all required skills
    for skill in required_skills:
        skill_mentions += len(re.findall(rf"\b{re.escape(skill)}\b", text))
        
    project_count = len(re.findall(r"project|developed|built|implemented", text))
    
    # Just a heuristic, can be adjusted
    if skill_mentions > 20 and project_count <= 2:
        return "High"
    elif skill_mentions > 15 and project_count <= 3:
        return "Medium"
    else:
        return "Low"

def detect_plagiarism_risk(text, required_skills):
    words = text.split()
    total_words = len(words)
    unique_ratio = len(set(words)) / total_words if total_words > 0 else 0
    
    buzz_count = 0 
    for skill in required_skills:
        buzz_count += text.count(skill)
        
    buzz_ratio = buzz_count / total_words if total_words > 0 else 0
    generic_phrases = len(re.findall(r"hardworking|dedicated|passionate|self-motivated|team player", text))
    
    if buzz_ratio > 0.05 or generic_phrases > 5 or unique_ratio < 0.4:
        return "High Risk"
    elif buzz_ratio > 0.03 or generic_phrases > 3:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------------- UI LAYOUT ----------------

# Icons (Sources: Icons8)
ICON_APP = "https://img.icons8.com/fluency/96/artificial-intelligence.png"
ICON_JOB = "https://img.icons8.com/fluency/48/briefcase.png"
ICON_CULTURE = "https://img.icons8.com/fluency/48/conference-call.png"
ICON_SETTINGS = "https://img.icons8.com/fluency/48/settings.png"
ICON_UPLOAD = "https://img.icons8.com/fluency/48/upload-to-cloud.png"
ICON_ACTION = "https://img.icons8.com/fluency/48/rocket.png"
ICON_TROPHY = "https://img.icons8.com/fluency/48/trophy.png"
ICON_MEDAL = "https://img.icons8.com/fluency/48/medal.png"

# Custom Header
st.markdown(f"""
    <div class="main-header">
        <img src="{ICON_APP}" alt="Logo">
        <h1>Smart HR Assistant</h1>
    </div>
    <div class="sub-header">Advanced AI-Powered Resume Screening & Parsing Engine</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(ICON_APP, width=80) 
    
    st.markdown(f"""
        <div class="feature-header">
            <img src="{ICON_JOB}">
            <h3>Job Configuration</h3>
        </div>
    """, unsafe_allow_html=True)
    
    job_description = st.text_area(
        "Job Description (JD)", 
        height=250, 
        placeholder="Paste the full job description here...",
        label_visibility="visible"
    )
    
    st.markdown(f"""
        <div class="feature-header" style="margin-top: 1rem;">
            <img src="{ICON_CULTURE}">
            <h3>Company Culture</h3>
        </div>
    """, unsafe_allow_html=True)

    company_culture = st.text_input(
        "Values", 
        placeholder="e.g. innovative, collaborative",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown(f"""
        <div class="feature-header">
            <img src="{ICON_SETTINGS}">
            <h3>Settings</h3>
        </div>
    """, unsafe_allow_html=True)
    
    shortlist_limit = st.slider("Candidates to Shortlist", 1, 50, 5)
    
    st.markdown("---")
    st.info("💡 **Tip:** Press 'R' to reload the app if it gets stuck.")

# Main Actions
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
        <div class="feature-header">
            <img src="{ICON_UPLOAD}">
            <h3>Upload Resumes</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drag and drop PDF resumes here", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col2:
    st.markdown(f"""
        <div class="feature-header">
            <img src="{ICON_ACTION}">
            <h3>Action</h3>
        </div>
    """, unsafe_allow_html=True)
    
    analyze_btn = st.button("Analyze Resumes", type="primary", use_container_width=True)
    if uploaded_files:
        st.caption(f"✅ {len(uploaded_files)} files selected")

# Logic
if analyze_btn:
    if not job_description:
        st.warning("⚠️ Please provide a **Job Description** in the sidebar to proceed.")
    elif not uploaded_files:
        st.warning("⚠️ Please **upload resumes** to analyze.")
    else:
        with st.spinner("🔄 AI is analyzing profiles..."):
            
            # Pre-calc JD requirements
            required_skills = extract_skills_from_jd(job_description.lower())
            desired_traits = extract_desired_traits(company_culture.lower())
            
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Processing {pdf_file.name}...")
                resume_text = extract_text_from_pdf(pdf_file)
                
                if not resume_text:
                    continue
                
                # Analysis
                skill_match = semantic_similarity(job_description.lower(), resume_text)
                matched_skills = skill_transparency(resume_text, required_skills)
                personality = ocean_personality_analysis(resume_text)
                culture_fit_score = calculate_culture_fit(personality, desired_traits)
                experience = experience_level(resume_text)
                project = project_match(resume_text)
                stuffing_risk = detect_skill_stuffing(resume_text, required_skills)
                plagiarism_risk = detect_plagiarism_risk(resume_text, required_skills)
                
                total_score = (skill_match * 0.6) + (culture_fit_score * 0.4)
                
                results.append({
                    "Candidate": pdf_file.name,
                    "Total Score": round(total_score, 2),
                    "Skill Match": round(skill_match, 2),
                    "Culture Fit": culture_fit_score,
                    "Experience": experience,
                    "Project Match": project,
                    "Matched Skills": ", ".join(matched_skills),
                    "Skill Count": len(matched_skills),
                    "Personality": ", ".join([k for k,v in personality.items() if v > 0]),
                    "Stuffing Risk": stuffing_risk,
                    "Plagiarism Risk": plagiarism_risk
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            
            # Sort results
            results = sorted(results, key=lambda x: x["Total Score"], reverse=True)
            df = pd.DataFrame(results)

            # ---------------- RESULTS DASHBOARD ----------------
            
            st.divider()
            
            # Top Stats
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidates Processed", len(results))
            m2.metric("Shortlisted", min(shortlist_limit, len(results)))
            m3.metric("Avg Match Score", f"{df['Total Score'].mean():.1f}%" if not df.empty else "0%")
            m4.metric("Top Score", f"{df['Total Score'].max():.1f}%" if not df.empty else "0%")
            
            st.markdown(f"""
                <div class="feature-header">
                    <img src="{ICON_TROPHY}">
                    <h3>Top Candidates</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Display best candidate in a special card
            if not df.empty:
                top_c = df.iloc[0]
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; align-items:center;">
                        <img src="{ICON_MEDAL}" style="width:40px; margin-right:15px;">
                        <div>
                            <h3 style="color: #4F46E5; margin:0;">Best Match: {top_c['Candidate']}</h3>
                            <p style="font-size: 1.1rem; margin:0;">Total Score: <b>{top_c['Total Score']}%</b> | Experience: <b>{top_c['Experience']}</b></p>
                        </div>
                    </div>
                    <p style="color: #9CA3AF; margin-top:10px;">Skills Found: {top_c['Matched Skills'][:150]}...</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Dataframe with nice column config
            st.markdown("#### Detailed Analysis")
            
            st.dataframe(
                df.head(shortlist_limit),
                use_container_width=True,
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Total Score",
                        help="Weighted score of Skills (60%) + Culture (40%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Skill Match": st.column_config.ProgressColumn(
                        "Skill Match",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Candidate": "Candidate Name",
                    "Stuffing Risk": st.column_config.TextColumn(
                        "Stuffing Risk",
                        help="Risk of keyword stuffing detected"
                    ),
                },
                hide_index=True
            )
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Report (CSV)",
                csv,
                f"smart_hr_report_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                type="secondary"
            )
