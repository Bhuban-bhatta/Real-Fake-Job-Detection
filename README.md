# 🔍 Real vs Fake Job Posting Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge&logo=xgboost"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/F1%20Score-0.84-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.92-brightgreen?style=for-the-badge"/>
</p>

A machine learning project that detects **fraudulent job postings** using Natural Language Processing (NLP) and classification algorithms. The model is trained on 17,880 real-world job postings and deployed via a Streamlit web application.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [EDA Insights](#-eda-insights)
- [Feature Engineering](#-feature-engineering)
- [Models Trained](#-models-trained)
- [Results](#-results)
- [Final Model](#-final-model)
- [Streamlit App](#-streamlit-app)
- [Installation](#-installation)
- [Usage](#-usage)

---

## 🎯 Problem Statement

Online job fraud is a growing threat — scammers post fake job listings to steal personal information or money from job seekers. This project builds a binary classifier to automatically detect fake job postings using text content and metadata signals.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [EMSCAD — Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| Total Records | 17,880 |
| Real Jobs | 17,014 (95.5%) |
| Fake Jobs | 866 (4.84%) |
| Class Imbalance | Severe — handled with SMOTE |

**Key Columns Used:**

| Column | Type | Use |
|---|---|---|
| `title` | Text | Job title |
| `company_profile` | Text | Company description |
| `description` | Text | Role details |
| `requirements` | Text | Skills needed |
| `required_education` | Text/Categorical | Education level |
| `required_experience` | Categorical | Experience level |
| `function` | Text | Job function |
| `has_company_logo` | Binary | Logo present? |
| `has_questions` | Binary | Screening questions? |
| `telecommuting` | Binary | Remote role? |
| `employment_type` | Categorical | Full-time/Part-time |

---

## 📁 Project Structure

```
Real-Fake-Job-Prediction/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── fake_job_posting.csv          # Original dataset
│   └── 📁 clean/
│       └── cleaned_data.csv              # Preprocessed dataset
│
├── 📁 notebook/
│   └── fake_job_detection.ipynb          # Full EDA + training notebook
│
├── 📁 src/
│   ├── preprocess.py                     # Text cleaning & feature engineering
│   ├── train.py                          # Model training
│   ├── prediction.py                     # Prediction pipeline
│   └── evaluate_and_save.py             # Evaluation & pkl export
│
├── 📁 images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance_xgboost.png
│   ├── feature_importance_randomforest.png
│   └── real_vs_fake_distribution.png
│
├── app.py                                # Streamlit web application
├── model.pkl                             # Saved XGBoost model
├── vectorizer.pkl                        # Saved TF-IDF vectorizer
├── numeric_cols.pkl                      # Saved numeric feature names
├── threshold.pkl                         # Saved decision threshold
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw CSV
   │
   ▼
Data Cleaning
  ├── Drop irrelevant columns (job_id, salary_range)
  ├── Remove 288 duplicates
  ├── Fill missing text with empty string
  ├── Remove HTML tags, URLs, special characters
  └── Fix data types
   │
   ▼
Data Leakage Check
  └── Remove rows containing leak keywords (fraud, scam) → 66 rows removed
   │
   ▼
Feature Engineering
  ├── Text: combine title + required_education + company_profile +
  │         description + requirements + required_experience + function
  ├── Length features: title_length, description_length,
  │                    requirements_length, company_profile_length
  ├── is_high_risk_education  (74.1% fake rate signal)
  ├── is_part_time            (9.3% fake rate signal)
  └── Text preprocessing: lowercase → tokenize → stopwords → stem
   │
   ▼
Train / Test Split (80/20, stratified)
   │
   ▼
TF-IDF Vectorization (max_features=5000, ngram_range=(1,2))
  └── fit on train only → transform test
   │
   ▼
Combine TF-IDF + Numeric → Feature Matrix (14020, 5009)
   │
   ▼
SMOTE (training data only)
  └── Real: 13,350 | Fake: 13,350
   │
   ▼
Model Training → Evaluation → Hyperparameter Tuning
   │
   ▼
Save Best Model (XGBoost) → Streamlit App
```

---

## 📈 EDA Insights

### Class Imbalance
- Only **4.84%** of postings are fake — severe imbalance handled with SMOTE

### Strongest Signals Found

| Feature | Finding |
|---|---|
| `has_company_logo = 0` | **15.9% fake rate** vs 2.0% with logo |
| `required_education = Some High School Coursework` | **74.1% fake rate** |
| `employment_type = Part-time` | **9.3% fake rate** |
| `telecommuting = 1` | **8.3% fake rate** |
| `company_profile length` | Real avg **636 chars** vs Fake avg **229 chars** (-407) |
| `requirements length` | Real avg **595 chars** vs Fake avg **444 chars** (-151) |

---

## ⚙️ Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `title_length` | Numeric | Character count of job title |
| `description_length` | Numeric | Character count of description |
| `requirements_length` | Numeric | Character count of requirements |
| `company_profile_length` | Numeric | Character count of company profile |
| `is_high_risk_education` | Binary | 1 if education level has >8% fake rate |
| `is_part_time` | Binary | 1 if employment type is Part-time |
| `has_company_logo` | Binary | Original — strongest single feature |
| `has_questions` | Binary | Original — screening question signal |
| `telecommuting` | Binary | Original — remote job signal |

---

## 🤖 Models Trained

| Model | Train F1 | Test F1 | Test AUC | Note |
|---|---|---|---|---|
| Logistic Regression | 0.9543 | 0.5677 | 0.8833 | High recall, low precision |
| RandomForestClassifier | 1.0000 | 0.7702 | 0.8209 | Overfit |
| XGBoost | 0.9987 | 0.7920 | 0.9186 | Best overall |
| MultinomialNB | — | — | — | TF-IDF only |

### After Hyperparameter Tuning

| Model | Best Params | Fake F1 | Fake Precision | Fake Recall |
|---|---|---|---|---|
| RandomForest | `class_weight={0:1,1:3}, n_estimators=200` | 0.75 | 0.94 | 0.62 |
| **XGBoost** | `max_depth=6, n_estimators=200, scale_pos_weight=10` | **0.84** | **0.88** | **0.80** |

### Voting Classifier (RF + XGBoost + LR)

| Metric | Score |
|---|---|
| F1 | 0.8258 |
| AUC | 0.9890 |
| Fake Precision | 0.86 |
| Fake Recall | 0.79 |

XGBoost outperformed Voting Classifier on F1 → **XGBoost selected as final model.**

---

## 📊 Results

### Numeric vs Text Features

| Feature Set | Test Accuracy |
|---|---|
| Numeric only | 96.35% |
| Text only | 97.92% |
| **Combined (Final)** | **98%** |

### Cross Validation (Correct — SMOTE inside folds)

| Model | CV F1 | CV AUC | CV Precision | CV Recall |
|---|---|---|---|---|
| Logistic Regression | 0.5728 | 0.9711 | 0.4287 | 0.8642 |
| RandomForest | 0.7448 | 0.9846 | 0.9408 | 0.6164 |
| **XGBoost** | **0.8170** | **0.9787** | **0.8500** | **0.7866** |

CV scores match test scores closely → **no data leakage, model is trustworthy.**

---

## 🏆 Final Model

**XGBoost Classifier** with tuned hyperparameters:

```python
XGBClassifier(
    max_depth        = 6,
    n_estimators     = 200,
    scale_pos_weight = 10,
    eval_metric      = 'logloss',
    random_state     = 42
)
```

### Final Classification Report

```
              precision    recall  f1-score   support

        Real       0.99      0.99      0.99      3321
        Fake       0.88      0.80      0.84       185

    accuracy                           0.98      3506
   macro avg       0.93      0.90      0.91      3506
weighted avg       0.98      0.98      0.98      3506
```

| Metric | Score |
|---|---|
| F1 (Fake) | **0.84** |
| Precision (Fake) | **0.88** |
| Recall (Fake) | **0.80** |
| ROC-AUC | **0.9186** |
| Overall Accuracy | **98%** |

---

## 🌐 Streamlit App

The app takes job posting details as input and predicts whether it is **Real or Fake**.

**Input Fields:**
- Job Title, Company Profile, Job Description, Requirements
- Required Education, Required Experience, Employment Type, Job Function
- Has Company Logo, Has Screening Questions, Telecommuting

**Output:**
- ✅ REAL or 🚨 FAKE prediction with confidence %
- Real / Fake probability metrics
- Red flag signals that triggered the prediction

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Real-Fake-Job-Prediction.git
cd Real-Fake-Job-Prediction

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
streamlit
numpy
pandas
scikit-learn
xgboost
imbalanced-learn
nltk
matplotlib
seaborn
wordcloud
```

---

## 🚀 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Run Prediction from Python

```python
import pickle
import numpy as np

tfidf        = pickle.load(open('vectorizer.pkl',   'rb'))
model        = pickle.load(open('model.pkl',        'rb'))
numeric_cols = pickle.load(open('numeric_cols.pkl', 'rb'))

# Build your feature vector and predict
prediction = model.predict(X)[0]
# 0 = Real, 1 = Fake
```

### Retrain the Model

```bash
# Run full notebook
jupyter notebook notebook/fake_job_detection.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| NLTK | Text preprocessing (tokenize, stem, stopwords) |
| Scikit-learn | ML models, TF-IDF, GridSearchCV, SMOTE pipeline |
| XGBoost | Final classifier |
| imbalanced-learn | SMOTE oversampling |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Web application |
| Pickle | Model serialization |

---

## 👤 Author

Made with ❤️ for fake job detection research.
