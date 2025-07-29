import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, matthews_corrcoef, roc_auc_score
)
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

# === Load and Prepare Data ===
df = pd.read_csv("covid_data.csv")
df['HasCases'] = df['daily_new_cases'].apply(lambda x: 1 if x > 0 else 0)
df = df.dropna()
df['country_encoded'] = LabelEncoder().fit_transform(df['country'])

features = ['country_encoded', 'cumulative_total_cases', 'active_cases', 'cumulative_total_deaths', 'daily_new_deaths']
X = df[features]
y = df['HasCases']
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Train Models ===
# 1. Baseline
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

# 2. Weighted
weighted_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
weighted_model.fit(X_train, y_train)
weighted_pred = weighted_model.predict(X_test)
weighted_prob = weighted_model.predict_proba(X_test)[:, 1]

# 3. Oversampled
def oversample(X, y):
    df_temp = pd.DataFrame(X)
    df_temp['target'] = y
    maj = df_temp[df_temp.target == 1]
    mino = df_temp[df_temp.target == 0]
    mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=42)
    df_upsampled = pd.concat([maj, mino_up])
    return df_upsampled.drop('target', axis=1).values, df_upsampled['target'].values

X_os, y_os = oversample(X_train, y_train)
oversampled_model = LogisticRegression(max_iter=1000, random_state=42)
oversampled_model.fit(X_os, y_os)
oversampled_pred = oversampled_model.predict(X_test)
oversampled_prob = oversampled_model.predict_proba(X_test)[:, 1]

# === Models Dictionary ===
models = {
    'Baseline LR': (baseline_pred, baseline_prob),
    'Weighted LR': (weighted_pred, weighted_prob),
    'Oversampled LR': (oversampled_pred, oversampled_prob)
}

# === Evaluation Summary Table ===
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL PERFORMANCE")
print("="*50)

for name, (pred, prob) in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)
    roc_auc = roc_auc_score(y_test, prob)
    
    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print(f"MCC            : {mcc:.4f}")

# === ROC Curves ===
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, (name, (_, prob)) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title('ROC Curves - Logistic Regression Model Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Confusion Matrix (Select One) ===
selected_model_name = 'Oversampled LR'
selected_pred = models[selected_model_name][0]

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, selected_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {selected_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# === Top 15 Countries by Cumulative Deaths ===
plt.figure(figsize=(12, 8))
top15 = df.groupby('country')['cumulative_total_deaths'].max().sort_values(ascending=False).head(15)
sns.barplot(x=top15.values, y=top15.index, palette='Reds_r', dodge=False)
plt.title("Top 15 Countries by Cumulative COVID-19 Deaths")
plt.xlabel("Cumulative Deaths")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# === Feature Importance (One Model) ===
selected_model_for_importance = baseline_model  # You can change to weighted_model or oversampled_model
feature_importance = pd.Series(selected_model_for_importance.coef_[0], index=features)
feature_importance = feature_importance.reindex(feature_importance.abs().sort_values(ascending=False).index)

plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar', color='skyblue')
plt.title("Feature Importance - Baseline Logistic Regression")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
