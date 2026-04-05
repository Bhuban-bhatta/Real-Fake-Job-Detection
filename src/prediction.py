# ── Load saved pkl files ───────────────────────────────────────────────────────
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tfidf        = pickle.load(open('vectorizer.pkl',   'rb'))
model        = pickle.load(open('model.pkl',        'rb'))
numeric_cols = pickle.load(open('numeric_cols.pkl', 'rb'))

ps        = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    if not text or str(text).strip() == '': return ''
    text   = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)

# ── Test Example 1 — Real Job ──────────────────────────────────────────────────
real_job = {
    'title'               : 'Software Engineer',
    'required_education'  : "Bachelor's Degree",
    'company_profile'     : 'Google is a multinational technology company specializing in internet services and products.',
    'description'         : 'We are looking for a skilled software engineer to join our team. You will design and develop scalable systems.',
    'requirements'        : 'Bachelor degree in Computer Science. 3+ years experience in Python or Java. Strong problem solving skills.',
    'required_experience' : 'Mid-Senior level',
    'function'            : 'Engineering',
    'has_company_logo'    : 1,
    'has_questions'       : 1,
    'telecommuting'       : 0,
    'employment_type'     : 'Full-time',
}

# ── Test Example 2 — Fake Job ──────────────────────────────────────────────────
fake_job = {
    'title'               : 'Work From Home Data Entry',
    'required_education'  : 'Some High School Coursework',
    'company_profile'     : '',
    'description'         : 'Earn up to $5000 weekly working from home. No experience needed. Easy work guaranteed income.',
    'requirements'        : '',
    'required_experience' : 'Not Applicable',
    'function'            : '',
    'has_company_logo'    : 0,
    'has_questions'       : 0,
    'telecommuting'       : 1,
    'employment_type'     : 'Part-time',
}

# ── Prediction Function ────────────────────────────────────────────────────────
def predict_job(job):
    # Combine text columns exactly as training
    combined = ' '.join([
        job['title'],
        job['required_education'],
        job['company_profile'],
        job['description'],
        job['requirements'],
        job['required_experience'],
        job['function'],
    ])

    # Preprocess and vectorize
    processed = transform_text(combined)
    text_vec  = tfidf.transform([processed]).toarray()

    # Build numeric features matching numeric_cols order
    numeric_map = {
        'has_company_logo'      : job['has_company_logo'],
        'has_questions'         : job['has_questions'],
        'telecommuting'         : job['telecommuting'],
        'title_length'          : len(job['title']),
        'description_length'    : len(job['description']),
        'requirements_length'   : len(job['requirements']),
        'company_profile_length': len(job['company_profile']),
        'is_high_risk_education': 1 if job['required_education'] in
                                  ['Some High School Coursework','Certification',
                                   'High School or equivalent'] else 0,
        'is_part_time'          : 1 if job['employment_type'] == 'Part-time' else 0,
    }
    num_vec = np.array([[numeric_map[c] for c in numeric_cols]])

    # Predict
    X           = np.hstack([text_vec, num_vec])
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    label = ' FAKE JOB' if prediction == 1 else 'REAL JOB'
    print(f'Title      : {job["title"]}')
    print(f'Prediction : {label}')
    print(f'Confidence : Real={probability[0]*100:.1f}% | Fake={probability[1]*100:.1f}%')
    print('-' * 50)

# ── Run Both Tests ─────────────────────────────────────────────────────────────
predict_job(real_job)
predict_job(fake_job)