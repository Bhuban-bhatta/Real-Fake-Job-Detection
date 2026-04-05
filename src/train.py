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


#data loaded

df = pd.read_csv('D:\data science mind riser\ML\Real-Fake-job-predection\Data\clean\cleaned_data.csv')
print(f'Loaded cleaned data: {df.shape}')

x_text    = df['transformed_text']
x_numeric = df[numeric_cols]
y         = df['fraudulent']

x_train_text, x_test_text, x_train_num, x_test_num, y_train, y_test = train_test_split(
    x_text, x_numeric, y, test_size=0.2, random_state=42
)
print(f'Train fake rate: {y_train.mean():.3f}')
print(f'Test  fake rate: {y_test.mean():.3f}')

#tf/idf vectorizer
# ngram_range=(1,2) captures fake phrases like 'work from home', 'earn money', 'no experience'
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

x_train_tfidf = tfidf.fit_transform(x_train_text)
x_test_tfidf  = tfidf.transform(x_test_text)
print(f'TF-IDF shape — Train: {x_train_tfidf.shape}, Test: {x_test_tfidf.shape}')


#combine numeric and tfidf feature
import numpy as np

x_train = np.hstack((x_train_tfidf.toarray(), x_train_num.values))
x_test  = np.hstack((x_test_tfidf.toarray(),  x_test_num.values))

print(f'Combined feature matrix shape — Train: {x_train.shape}, Test: {x_test.shape}')

#apply smote
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

# Save original y_train before any resampling
y_train_original = y_train.copy()

# Group 1 — combined features
x_train, y_train = smote.fit_resample(x_train, y_train_original)

# Group 2 — tfidf only, use original so sizes match
x_train_text, _ = smote.fit_resample(x_train_tfidf, y_train_original)

print('After SMOTE:')
print(pd.Series(y_train).value_counts())

#train the model


models = {
    'Logistic Regression'   : LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBoost'               : XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=20, random_state=42,
        eval_metric='logloss', verbosity=0
    )
}


results = []

# Train
for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred  = model.predict(x_test)
    results.append({
        'Model'          : model_name,
        'Train Accuracy' : accuracy_score(y_train, y_train_pred),
        'Train Precision': precision_score(y_train, y_train_pred),
        'Train Recall'   : recall_score(y_train, y_train_pred),
        'Train F1'       : f1_score(y_train, y_train_pred),
        'Train ROC-AUC'  : roc_auc_score(y_train, y_train_pred),
        'Test Accuracy'  : accuracy_score(y_test, y_test_pred),
        'Test Precision' : precision_score(y_test, y_test_pred),
        'Test Recall'    : recall_score(y_test, y_test_pred),
        'Test F1'        : f1_score(y_test, y_test_pred),
        'Test ROC-AUC'   : roc_auc_score(y_test, y_test_pred),
    })

# Display combined results
results_df = pd.DataFrame(results).set_index('Model').round(4)
display(results_df)