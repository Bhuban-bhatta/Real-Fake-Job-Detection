# src/evaluate_and_save.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.parse_math'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import os

# ── Import artifacts from train.py ────────────────────────────────────────────
from train import TRAIN_ARTIFACTS

models           = TRAIN_ARTIFACTS['models']
nb_model         = TRAIN_ARTIFACTS['nb_model']
best_xgb         = TRAIN_ARTIFACTS['best_xgb']
x_test           = TRAIN_ARTIFACTS['x_test']
x_test_tfidf     = TRAIN_ARTIFACTS['x_test_tfidf']
x_test_num       = TRAIN_ARTIFACTS['x_test_num']
y_test           = TRAIN_ARTIFACTS['y_test']
y_train          = TRAIN_ARTIFACTS['y_train']
y_train_original = TRAIN_ARTIFACTS['y_train_original']
x_train          = TRAIN_ARTIFACTS['x_train']
tfidf            = TRAIN_ARTIFACTS['tfidf']
numeric_cols     = TRAIN_ARTIFACTS['numeric_cols']

os.makedirs('images', exist_ok=True)

# ── 1. Model results table ─────────────────────────────────────────────────────
results = []

for model_name, model in models.items():
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

for model_name, model in nb_model.items():
    y_train_pred = model.predict(TRAIN_ARTIFACTS['x_test_tfidf'])
    y_test_pred  = model.predict(x_test_tfidf)
    results.append({
        'Model'          : model_name,
        'Train Accuracy' : accuracy_score(y_test, y_test_pred),
        'Train Precision': precision_score(y_test, y_test_pred),
        'Train Recall'   : recall_score(y_test, y_test_pred),
        'Train F1'       : f1_score(y_test, y_test_pred),
        'Train ROC-AUC'  : roc_auc_score(y_test, y_test_pred),
        'Test Accuracy'  : accuracy_score(y_test, y_test_pred),
        'Test Precision' : precision_score(y_test, y_test_pred),
        'Test Recall'    : recall_score(y_test, y_test_pred),
        'Test F1'        : f1_score(y_test, y_test_pred),
        'Test ROC-AUC'   : roc_auc_score(y_test, y_test_pred),
    })

results_df = pd.DataFrame(results).set_index('Model').round(4)
print('\n--- Model Results ---')
print(results_df)

# ── 2. Classification reports — top 3 models ──────────────────────────────────
best_models = results_df.nlargest(3, 'Test F1').index
for name in best_models:
    model  = models[name]
    y_pred = model.predict(x_test)
    print(f"{'='*50}")
    print(f'  {name}')
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# ── 3. Confusion matrices ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, model) in zip(axes.flat, models.items()):
    y_pred = model.predict(x_test)
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=['Real', 'Fake']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name} | F1={f1_score(y_test, y_pred):.3f}')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
print(' Saved: images/confusion_matrix.png')

# ── 4. ROC curves ─────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 7))
for (name, model), color in zip(models.items(), plt.cm.tab10(np.linspace(0, 1, len(models)))):
    y_prob       = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f'{name} (AUC={roc_auc_score(y_test, y_prob):.4f})')

plt.plot([0,1],[0,1],'--', color='gray', lw=1, alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('images/roc_curve.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ Saved: images/roc_curve.png')

# ── 5. Feature importance — RandomForest ──────────────────────────────────────
rf_model   = models['RandomForestClassifier']
feat_names = list(tfidf.get_feature_names_out()) + list(numeric_cols)
top_idx    = np.argsort(rf_model.feature_importances_)[::-1][:25]
colors     = ['#E05C5C' if f in numeric_cols else '#4A90D9' for f in np.array(feat_names)[top_idx]]

plt.figure(figsize=(10, 7))
plt.barh(range(25), rf_model.feature_importances_[top_idx][::-1], color=colors[::-1])
plt.yticks(range(25), np.array(feat_names)[top_idx[::-1]], fontsize=9)
plt.title('Top 25 Feature Importances — RandomForestClassifier')
plt.tight_layout()
plt.savefig('images/feature_importance_randomforest.png', dpi=120, bbox_inches='tight')
plt.show()
print(' Saved: images/feature_importance_randomforest.png')

# ── 6. Feature importance — XGBoost ──────────────────────────────────────────
top_idx = np.argsort(best_xgb.feature_importances_)[::-1][:25]
colors  = ['#E05C5C' if f in numeric_cols else '#4A90D9' for f in np.array(feat_names)[top_idx]]

plt.figure(figsize=(10, 7))
plt.barh(range(25), best_xgb.feature_importances_[top_idx][::-1], color=colors[::-1])
plt.yticks(range(25), np.array(feat_names)[top_idx[::-1]], fontsize=9)
plt.title('Top 25 Feature Importances — XGBoost')
plt.tight_layout()
plt.savefig('images/feature_importance_xgboost.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved: images/feature_importance_xgboost.png')

# ── 7. Real vs Fake distribution ──────────────────────────────────────────────
import pandas as pd
df = pd.read_csv('data/clean/cleaned_data.csv')
ax = df['fraudulent'].value_counts().plot(
    kind='bar', color=['steelblue','crimson'],
    figsize=(6, 4), edgecolor='white'
)
ax.set_xticklabels(['Real', 'Fake'], rotation=0)
ax.set_title('Real vs Fake Job Distribution')
ax.set_ylabel('Count')
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f'{int(bar.get_height()):,}',
            ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('images/real_vs_fake_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved: images/real_vs_fake_distribution.png')

print('\n All evaluation complete — images saved to images/')
