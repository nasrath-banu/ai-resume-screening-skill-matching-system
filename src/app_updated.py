import streamlit as st
import pdfplumber
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Utility Functions
# ---------------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(p.text for p in doc.paragraphs).lower()

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text)

def extract_skills(text, skill_list):
    return sorted(set(skill for skill in skill_list if skill in text))

def skill_match_score(matched, jd_skills):
    if not jd_skills:
        return 0
    return round((len(matched) / len(jd_skills)) * 100, 2)

def tfidf_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def hybrid_score(skill_score, tfidf_score, w1=0.7, w2=0.3):
    return round((skill_score * w1) + (tfidf_score * w2), 2)

# ---------------------------------
# Role-Based Skill Database
# ---------------------------------

ROLE_SKILLS = {
    "data analyst": [
        "python", "sql", "excel", "power bi", "tableau",
        "numpy", "pandas", "statistics", "data visualization","data mining"
    ],
    "web developer": [
        "html", "css", "javascript", "react",
        "node", "mongodb", "express"
    ],
    "machine learning engineer": [
        "python", "numpy", "pandas",
        "scikit learn", "tensorflow", "machine learning","pytorch"
        ,"docker","deep learning","NLP","tensorflow"
    ],
    "ai enginner":[
        "generative ai","python","java","c++","advanced mathematics","LLM","prompt engineering","langchain","cnn","rnn","gan"
    ],
    "data scientist":[
        "statistics","python","machine learning","regression","classification","nlp","xgboost","power bi","seaborn","matplotlib","hadoop"
    ]
}

def detect_role(jd_text):
    scores = {}
    for role, skills in ROLE_SKILLS.items():
        scores[role] = sum(skill in jd_text for skill in skills)
    return max(scores, key=scores.get)


# ---------------------------------
# Streamlit UI
# ---------------------------------

st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("AI Resume Screening & Skill Matching System")
st.write("Upload a resume and paste the job description to get match insights.")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)", type=["pdf", "docx"]
)

jd_text = st.text_area("Paste Job Description Here", height=260)

if uploaded_file and jd_text:
    st.success("Inputs received. Ready for analysis.")

    if st.button("Analyze Resume"):

        # Resume Extraction
        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

        resume_text = clean_text(resume_text)
        jd_text_clean = clean_text(jd_text.lower())

        # Role Detection
        detected_role = detect_role(jd_text_clean)
        active_skills = ROLE_SKILLS[detected_role]

        st.info(f"Detected Job Role: {detected_role.title()}")

        # Skill Extraction
        resume_skills = extract_skills(resume_text, active_skills)
        jd_skills = extract_skills(jd_text_clean, active_skills)

        matched_skills = list(set(resume_skills).intersection(set(jd_skills)))
        missing_skills = list(set(jd_skills) - set(resume_skills))

        skill_score = skill_match_score(matched_skills, jd_skills)
        tfidf_score = tfidf_similarity(resume_text, jd_text_clean)
        final_score = hybrid_score(skill_score, tfidf_score)

        # ---------------------------------
        # Results Section
        # ---------------------------------

        st.subheader("Match Scores")

        c1, c2, c3 = st.columns(3)
        c1.metric("Skill Match", f"{skill_score}%")
        c2.metric("Text Similarity", f"{tfidf_score}%")
        c3.metric("Hybrid Score", f"{final_score}%")

        st.subheader("Final Evaluation")
        st.progress(final_score / 100)

        if final_score >= 75:
            st.success(f"Strong Match – {final_score}%")
        elif final_score >= 50:
            st.warning(f"Moderate Match – {final_score}%")
        else:
            st.error(f"Low Match – {final_score}%")

        # ---------------------------------
        # Skill Analysis
        # ---------------------------------

        st.subheader("Skill Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matched Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f"- {skill.title()}")
            else:
                st.write("No matched skills found")

        with col2:
            st.markdown("### ❌ Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f"- {skill.title()}")
            else:
                st.write("No missing skills")

        # ---------------------------------
        # Recommendation
        # ---------------------------------

        if missing_skills:
            st.info(
                "📌 **Recommended skills to improve:** "
                + ", ".join(skill.title() for skill in missing_skills)
            )
