
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
matplotlib_inline = plt.rcParams.update({'text.parse_math': False})
SEED = 42
print(' All imports successful')

df = pd.read_csv('fake_job_postings.csv')
print(df.shape)
df.head(2)

df.info()

df.isnull().sum()

df.describe()

# Drop irrelevant columns
df.drop(columns=['job_id', 'salary_range'], inplace=True)
print('Dropped: job_id, salary_range')
print('Shape after drop:', df.shape)

# Check and remove duplicates
print('Duplicates before:', df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print('Duplicates after:', df.duplicated().sum())
print('Shape after dedup:', df.shape)

# Fill missing values in text columns with empty string
text_fill_cols = [
    'title', 'location', 'department', 'company_profile',
    'description', 'requirements', 'benefits',
    'employment_type', 'required_experience',
    'required_education', 'industry', 'function'
]
df[text_fill_cols] = df[text_fill_cols].fillna('')
print('Missing text values filled with empty string')
print(df.isnull().sum())

# Clean text — remove HTML tags, URLs, special characters from key text columns
def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)        # remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'&amp;|&lt;|&gt;', ' ', text) # remove HTML entities
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
    df[col] = df[col].apply(clean_text)

print('HTML tags, URLs, special characters removed')
print('Sample cleaned description:')
print(df['description'].iloc[0][:200])

# Fix data types for numeric columns
df['telecommuting']    = df['telecommuting'].astype(int)
df['has_company_logo'] = df['has_company_logo'].astype(int)
df['has_questions']    = df['has_questions'].astype(int)
df['fraudulent']       = df['fraudulent'].astype(int)
print('Data types fixed')
print(df[['telecommuting','has_company_logo','has_questions','fraudulent']].dtypes)

#save clean data
df.to_csv("cleaned_data.csv", index=False)

"""EDA


"""

ax = sns.countplot(data=df, x='fraudulent', hue='fraudulent', palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Distribution of Target: fraudulent')
plt.show()

ax = sns.countplot(data=df, x='has_company_logo', hue='fraudulent', palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('has_company_logo vs fraudulent')
plt.show()
print(df.groupby('has_company_logo')['fraudulent'].mean().round(3))

ax = sns.countplot(data=df, x='has_questions', hue='fraudulent', palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('has_questions vs fraudulent')
plt.show()
print(df.groupby('has_questions')['fraudulent'].mean().round(3))

ax = sns.countplot(data=df, x='telecommuting', hue='fraudulent', palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('telecommuting vs fraudulent')
plt.show()
print(df.groupby('telecommuting')['fraudulent'].mean().round(3))

print(df.groupby('employment_type')['fraudulent'].mean().round(3).sort_values(ascending=False))
ax = sns.countplot(data=df, x='employment_type', hue='fraudulent', palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('employment_type vs fraudulent')
plt.xticks(rotation=15)
plt.show()

print(df.groupby('required_education')['fraudulent'].mean().round(3).sort_values(ascending=False))

print(df.groupby('required_experience')['fraudulent'].mean().round(3).sort_values(ascending=False))

# Text length comparison Real vs Fake
for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
    real_len = df[df.fraudulent==0][col].fillna('').str.len().mean()
    fake_len = df[df.fraudulent==1][col].fillna('').str.len().mean()
    print(f'{col:<20} Real={real_len:.0f} | Fake={fake_len:.0f} | Diff={fake_len-real_len:.0f}')

# Check for leakage keywords and remove those rows
leak_words = ['fake', 'fraud', 'scam', 'fraudulent']
rows_to_drop = pd.Index([])

for word in leak_words:
    count = df[df['fraudulent']==1]['description'].fillna('').str.contains(word, case=False).sum()
    print(f"'{word}' appears in {count} fake job texts")
    mask = df['description'].fillna('').str.contains(word, case=False, na=False)
    rows_to_drop = rows_to_drop.union(df[mask].index)

df = df.drop(index=rows_to_drop).reset_index(drop=True)
print(f'\nRemoved {len(rows_to_drop)} leakage rows. New shape: {df.shape}')

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    if not text or str(text).strip() == '': return ''
    text   = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)

# Fill missing values
text_cols_to_fill = ['title','required_education','company_profile','description',
                     'requirements','required_experience','function']
df[text_cols_to_fill] = df[text_cols_to_fill].fillna('')

# Combine text columns (benefits excluded — no signal difference between real/fake)
cols = [
    'title',
    'required_education',
    'company_profile',
    'description',
    'requirements',
    'required_experience',
    'function',
]
df['text'] = df[cols].fillna('').agg(' '.join, axis=1)
df['transformed_text'] = df['text'].apply(transform_text)
print('Text preprocessing done')
df['transformed_text'].head(2)

# Length features — fake jobs have much shorter text in these columns
df['title_length']           = df['title'].str.len()
df['description_length']     = df['description'].str.len()
df['requirements_length']    = df['requirements'].str.len()
df['company_profile_length'] = df['company_profile'].str.len()

# High risk education — 74.1% fake rate for 'Some High School Coursework'
high_risk_edu = ['Some High School Coursework', 'Certification', 'High School or equivalent']
df['is_high_risk_education'] = df['required_education'].isin(high_risk_edu).astype(int)

# Part time flag — 9.3% fake rate
df['is_part_time'] = (df['employment_type'] == 'Part-time').astype(int)


print('Feature engineering done')
print(df[['title_length','description_length','requirements_length',
          'company_profile_length','is_high_risk_education',
          'is_part_time']].head(3))

#combine all numeric features
numeric_cols = [
    'has_company_logo',
    'has_questions',
    'telecommuting',
    'title_length',
    'description_length',
    'requirements_length',
    'company_profile_length',
    'is_high_risk_education',
    'is_part_time'
]

from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['fraudulent']==1]['transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)
plt.axis('off')
plt.title('Fake Jobs — Word Cloud')
plt.show()

ham_wc = wc.generate(df[df['fraudulent']==0]['transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)
plt.axis('off')
plt.title('Real Jobs — Word Cloud')
plt.show()

