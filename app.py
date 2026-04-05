import streamlit as st
import pickle
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Fake Job Detector", page_icon="🔍", layout="wide")

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

@st.cache_resource
def load_artifacts():
    vectorizer   = pickle.load(open('vectorizer.pkl',   'rb'))
    model        = pickle.load(open('model.pkl',        'rb'))
    numeric_cols = pickle.load(open('numeric_cols.pkl', 'rb'))
    return vectorizer, model, numeric_cols

vectorizer, model, numeric_cols = load_artifacts()

ps        = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    if not text or str(text).strip() == '': return ''
    text   = re.sub(r'<[^>]+>', ' ', str(text))
    text   = re.sub(r'http\S+', '', text)
    text   = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)

HIGH_RISK_EDU = ['Some High School Coursework', 'Certification', 'High School or equivalent']

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔍 Fake Job Posting Detector")
st.markdown("Enter job details below to check if a posting is likely **real or fake**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Text")
    title               = st.text_input("Job Title")
    company_profile     = st.text_area("Company Profile",  height=80)
    description         = st.text_area("Job Description",  height=120)
    requirements        = st.text_area("Requirements",      height=100)

with col2:
    st.subheader("🏢 Company & Job Metadata")
    required_education  = st.selectbox("Required Education", [
        "", "Bachelor's Degree", "Master's Degree",
        "High School or equivalent", "Some College Coursework Completed",
        "Associate Degree", "Professional", "Doctorate",
        "Certification", "Unspecified", "Some High School Coursework"
    ])
    required_experience = st.selectbox("Required Experience", [
        "", "Not Applicable", "Internship", "Entry level",
        "Associate", "Mid-Senior level", "Director", "Executive"
    ])
    employment_type     = st.selectbox("Employment Type", [
        "Full-time", "Part-time", "Contract", "Temporary", "Other"
    ])
    job_function        = st.text_input("Job Function", placeholder="e.g. Engineering, Sales")
    has_company_logo    = st.radio("Company has a logo?",        [1, 0], format_func=lambda x: "Yes" if x else "No")
    has_questions       = st.radio("Has screening questions?",   [1, 0], format_func=lambda x: "Yes" if x else "No")
    telecommuting       = st.radio("Remote / Work from home?",   [0, 1], format_func=lambda x: "Yes" if x else "No")

if st.button("🔍 Predict", type="primary", use_container_width=True):
    # Combine text columns exactly as training
    combined_text = " ".join([
        title, required_education, company_profile,
        description, requirements, required_experience, job_function
    ])
    processed = transform_text(combined_text)
    text_vec  = vectorizer.transform([processed]).toarray()

    # Build numeric features matching numeric_cols order from pkl
    numeric_feat = {
        'has_company_logo'      : has_company_logo,
        'has_questions'         : has_questions,
        'telecommuting'         : telecommuting,
        'title_length'          : len(title),
        'description_length'    : len(description),
        'requirements_length'   : len(requirements),
        'company_profile_length': len(company_profile),
        'is_high_risk_education': 1 if required_education in HIGH_RISK_EDU else 0,
        'is_part_time'          : 1 if employment_type == 'Part-time' else 0,
    }
    num_vec    = np.array([[numeric_feat[c] for c in numeric_cols]])
    X          = np.hstack([text_vec, num_vec])

    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **FAKE** — Likely fake (confidence: {probability[1]*100:.1f}%)")
    else:
        st.success(f"✅ **REAL** — Appears legitimate (confidence: {probability[0]*100:.1f}%)")

    col_r, col_f = st.columns(2)
    col_r.metric("Real Probability", f"{probability[0]*100:.1f}%")
    col_f.metric("Fake Probability", f"{probability[1]*100:.1f}%")

    with st.expander("🔎 What signals triggered this prediction?"):
        signals = []
        if not has_company_logo:                        signals.append("⚠️ No company logo")
        if not has_questions:                           signals.append("⚠️ No screening questions")
        if len(company_profile.strip()) == 0:           signals.append("⚠️ No company profile provided")
        if len(description.split()) < 30:               signals.append("⚠️ Very short job description")
        if len(requirements.strip()) == 0:              signals.append("⚠️ No requirements listed")
        if required_education in HIGH_RISK_EDU:         signals.append("⚠️ High-risk education level (74% fake rate)")
        if employment_type == 'Part-time':              signals.append("⚠️ Part-time role (9.3% fake rate)")
        if telecommuting:                               signals.append("⚠️ Remote/telecommuting role (8.3% fake rate)")
        for s in signals: st.write(s)
        if not signals: st.write("✅ No major red flags detected")
