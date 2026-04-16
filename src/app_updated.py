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
        "numpy", "pandas", "statistics", "data visualization", "data mining",
        "mysql", "postgresql", "r", "looker", "google analytics",
        "etl", "data cleaning", "pivot tables", "vlookup", "dashboard",
        "business intelligence", "reporting", "data wrangling", "snowflake", "spark"
    ],
    "web developer": [
        "html", "css", "javascript", "react", "node", "mongodb", "express",
        "typescript", "angular", "vue", "nextjs", "rest api", "graphql",
        "git", "github", "bootstrap", "tailwind", "php", "mysql",
        "redux", "webpack", "docker", "aws", "firebase", "responsive design"
    ],
    "machine learning engineer": [
        "python", "numpy", "pandas", "scikit learn", "tensorflow",
        "machine learning", "pytorch", "docker", "deep learning", "nlp",
        "xgboost", "lightgbm", "keras", "mlops", "feature engineering",
        "model deployment", "aws", "azure", "pyspark", "computer vision",
        "regression", "classification", "clustering", "random forest", "neural networks"
    ],
    "ai engineer": [
        "generative ai", "python", "java", "c++", "advanced mathematics",
        "llm", "prompt engineering", "langchain", "cnn", "rnn", "gan",
        "openai", "hugging face", "transformer", "bert", "gpt",
        "vector database", "rag", "fine tuning", "embeddings",
        "fastapi", "mlflow", "reinforcement learning", "multimodal ai", "llama"
    ],
    "data scientist": [
        "statistics", "python", "machine learning", "regression", "classification",
        "nlp", "xgboost", "power bi", "seaborn", "matplotlib", "hadoop",
        "numpy", "pandas", "scikit learn", "deep learning", "tensorflow",
        "pytorch", "sql", "r", "hypothesis testing", "a b testing",
        "feature engineering", "data wrangling", "pyspark", "tableau", "clustering"
    ],
    "software engineer": [
        "python", "java", "c++", "c#", "golang", "rust",
        "data structures", "algorithms", "rest api", "microservices",
        "spring boot", "django", "fastapi", "postgresql", "redis",
        "docker", "kubernetes", "git", "ci cd", "unit testing",
        "system design", "oop", "linux", "kafka", "rabbitmq"
    ],
    "devops engineer": [
        "docker", "kubernetes", "jenkins", "git", "github actions",
        "aws", "azure", "gcp", "terraform", "ansible",
        "linux", "bash", "ci cd", "monitoring", "prometheus",
        "grafana", "nginx", "helm", "cloudformation", "argocd",
        "security", "networking", "python", "elk stack", "vault"
    ],
    "cybersecurity analyst": [
        "network security", "penetration testing", "ethical hacking", "siem",
        "firewalls", "vulnerability assessment", "python", "linux",
        "wireshark", "metasploit", "nmap", "ids ips", "encryption",
        "iso 27001", "owasp", "soc", "threat analysis", "incident response",
        "splunk", "burp suite", "kali linux", "zero trust", "malware analysis"
    ],
    "business analyst": [
        "requirement gathering", "sql", "excel", "power bi", "tableau",
        "stakeholder management", "brd", "uml", "jira", "confluence",
        "agile", "scrum", "data analysis", "process mapping", "wireframing",
        "business intelligence", "gap analysis", "user stories", "ms visio", "reporting"
    ],
    "ui ux designer": [
        "figma", "adobe xd", "sketch", "prototyping", "wireframing",
        "user research", "usability testing", "information architecture",
        "interaction design", "typography", "color theory", "design systems",
        "invision", "zeplin", "html", "css", "accessibility",
        "user journey", "a b testing", "responsive design"
    ],
    "product manager": [
        "product roadmap", "agile", "scrum", "jira", "confluence",
        "stakeholder management", "user stories", "market research",
        "product strategy", "kpi", "okr", "a b testing", "sql",
        "data analysis", "wireframing", "figma", "go to market",
        "competitive analysis", "prioritization", "product lifecycle"
    ],
    "database administrator": [
        "sql", "mysql", "postgresql", "oracle", "sql server",
        "mongodb", "redis", "database design", "indexing", "query optimization",
        "backup recovery", "replication", "partitioning", "nosql", "cassandra",
        "performance tuning", "stored procedures", "triggers", "etl", "data warehousing"
    ],
    "full stack developer": [
        "html", "css", "javascript", "react", "angular", "vue",
        "node", "express", "python", "django", "rest api", "graphql",
        "mongodb", "postgresql", "mysql", "docker", "git", "aws",
        "typescript", "redux", "nextjs", "ci cd", "microservices",
        "tailwind", "bootstrap", "firebase", "linux", "nginx"
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
