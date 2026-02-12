import os
import re
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- HR INPUT ----------------

JOB_DESCRIPTION = input("\nPaste Job Description:\n").lower()
COMPANY_CULTURE = input("\nEnter culture / values of the company:\n").lower()

RESUME_FOLDER = r"C:\Users\Dell\Downloads\archive (19)\CVs1\CVs1"

# ---------------- PDF EXTRACTION (Improved) ----------------

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
        text = re.sub(r"\s+", " ", text)
        return text.lower()
    except:
        return ""

# ---------------- SEMANTIC MODEL ----------------

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100

# ---------------- DYNAMIC SKILL EXTRACTION ----------------

def extract_skills_from_jd(job_desc):
    words = re.findall(r"\b[A-Za-z\+\#\.]+\b", job_desc)
    stopwords = {"with","and","for","the","looking","experience","engineer","developer","required"}
    skills = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
    return list(set(skills))

REQUIRED_SKILLS = extract_skills_from_jd(JOB_DESCRIPTION)

def skill_transparency(resume_text, required_skills):
    matched = []
    for skill in required_skills:
        if re.search(rf"\b{re.escape(skill)}\b", resume_text):
            matched.append(skill)
    return matched

# ---------------- OCEAN PERSONALITY (Improved) ----------------

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

# ---------------- CULTURE TRAIT EXTRACTION ----------------

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

DESIRED_TRAITS = extract_desired_traits(COMPANY_CULTURE)

def calculate_culture_fit(personality, desired_traits):
    return sum(personality.get(trait, 0) for trait in desired_traits)

# ---------------- ADVANCED EXPERIENCE DETECTION ----------------

def experience_level(text):

    numeric_match = re.findall(r"(\d+)\+?\s*(years|yrs)", text)
    for match in numeric_match:
        years = int(match[0])
        if years >= 5:
            return "High"
        elif years >= 3:
            return "Medium"
        else:
            return "Low"

    written_years = {
        "five":5,"six":6,"seven":7,
        "eight":8,"nine":9,"ten":10
    }

    for word,value in written_years.items():
        if re.search(rf"{word}\s+years", text):
            if value >= 5:
                return "High"
            elif value >= 3:
                return "Medium"

    date_ranges = re.findall(r"(20\d{2})\s*[-–]\s*(20\d{2}|present)", text)

    for start,end in date_ranges:
        start = int(start)
        end = datetime.now().year if end == "present" else int(end)
        diff = end - start
        if diff >= 5:
            return "High"
        elif diff >= 3:
            return "Medium"

    return "Low"

# ---------------- PROJECT MATCH ----------------

def project_match(text):
    return "High" if re.search(r"project|developed|built|implemented", text) else "Low"

# ---------------- SKILL STUFFING DETECTION ----------------

def detect_skill_stuffing(text):
    skill_mentions = len(REQUIRED_SKILLS)
    project_count = len(re.findall(r"project|developed|built|implemented", text))

    if skill_mentions > 10 and project_count <= 2:
        return "High"
    elif skill_mentions > 7 and project_count <= 3:
        return "Medium"
    else:
        return "Low"

# ---------------- INTERNAL PLAGIARISM RISK ----------------

def detect_plagiarism_risk(text):
    words = text.split()
    total_words = len(words)
    unique_ratio = len(set(words)) / total_words if total_words > 0 else 0

    buzz_count = sum(text.count(skill) for skill in REQUIRED_SKILLS)
    buzz_ratio = buzz_count / total_words if total_words > 0 else 0

    generic_phrases = len(re.findall(r"hardworking|dedicated|passionate|self-motivated|team player", text))

    if buzz_ratio > 0.05 or generic_phrases > 5 or unique_ratio < 0.4:
        return "High Risk"
    elif buzz_ratio > 0.03 or generic_phrases > 3:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------------- LOAD FILES ----------------

pdf_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]

if len(pdf_files) == 0:
    print("No resume files found!")
    exit()

print(f"\n{len(pdf_files)} resume files found!\n")

# ---------------- PROCESS ----------------

results = []

for pdf in pdf_files:

    resume_text = extract_text_from_pdf(os.path.join(RESUME_FOLDER, pdf))

    skill_match = semantic_similarity(JOB_DESCRIPTION, resume_text)
    matched_skills = skill_transparency(resume_text, REQUIRED_SKILLS)

    personality = ocean_personality_analysis(resume_text)
    culture_fit = calculate_culture_fit(personality, DESIRED_TRAITS)

    experience = experience_level(resume_text)
    project = project_match(resume_text)

    stuffing_risk = detect_skill_stuffing(resume_text)
    plagiarism_risk = detect_plagiarism_risk(resume_text)

    total_score = (skill_match * 0.6) + (culture_fit * 0.4)

    results.append({
        "Candidate": pdf.replace(".pdf",""),
        "Skill Match": int(skill_match),
        "Matched Skills": matched_skills,
        "Culture Fit": culture_fit,
        "Personality": [t for t,v in personality.items() if v>0],
        "Experience Level": experience,
        "Project Match": project,
        "Skill Stuffing Risk": stuffing_risk,
        "Plagiarism Risk": plagiarism_risk,
        "Total Score": int(total_score)
    })

# ---------------- SORT ----------------

results = sorted(results, key=lambda x: x["Total Score"], reverse=True)

# ---------------- SHORTLIST ----------------

shortlist_count = int(input("Enter number of candidates to shortlist: "))
shortlist_count = min(shortlist_count, len(results))

top_candidates = results[:shortlist_count]

# ---------------- OUTPUT ----------------

print(f"\n================ TOP {shortlist_count} CANDIDATES ================\n")

for rank, candidate in enumerate(top_candidates, 1):
    print(f"Rank {rank}: {candidate['Candidate']}")
    print(f"Skill Match        : {candidate['Skill Match']}%")
    print(f"Matched Skills     : {candidate['Matched Skills']}")
    print(f"Culture Fit        : {candidate['Culture Fit']}%")
    print(f"Personality        : {candidate['Personality']}")
    print(f"Experience Level   : {candidate['Experience Level']}")
    print(f"Project Match      : {candidate['Project Match']}")
    print(f"Skill Stuffing Risk: {candidate['Skill Stuffing Risk']}")
    print(f"Plagiarism Risk    : {candidate['Plagiarism Risk']}")
    print(f"Total Score        : {candidate['Total Score']}%")
    print("--------------------------------------------------")

























