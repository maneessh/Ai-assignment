import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, matthews_corrcoef,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load dataset
print("Loading COVID-19 dataset...")
df = pd.read_csv("covid_data.csv")

# Basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Create binary target: 1 if daily_new_cases > 0 else 0
df['HasCases'] = df['daily_new_cases'].apply(lambda x: 1 if x > 0 else 0)

# Check class distribution before preprocessing
print(f"\nOriginal class distribution:")
print(df['HasCases'].value_counts())
print(f"Class ratio (1:0): {df['HasCases'].value_counts()[1] / df['HasCases'].value_counts()[0]:.2f}")

# Drop missing values
df = df.dropna()
print(f"After removing missing values: {df.shape}")

# Encode categorical 'country'
le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])

# Features and target
features = ['country_encoded', 'cumulative_total_cases', 'active_cases', 'cumulative_total_deaths', 'daily_new_deaths']
X = df[features]
y = df['HasCases']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set class distribution:")
print(pd.Series(y_train).value_counts())
print(f"Test set class distribution:")
print(pd.Series(y_test).value_counts())

# ===== MODEL 1: BASELINE LOGISTIC REGRESSION =====
print("\n" + "="*50)
print("MODEL 1: BASELINE LOGISTIC REGRESSION")
print("="*50)

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

print("Baseline Model Results:")
print(f"Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
print(f"Precision: {precision_score(y_test, baseline_pred):.4f}")
print(f"Recall: {recall_score(y_test, baseline_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, baseline_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, baseline_prob):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, baseline_pred):.4f}")

# ===== MODEL 2: CLASS WEIGHTED LOGISTIC REGRESSION =====
print("\n" + "="*50)
print("MODEL 2: CLASS WEIGHTED LOGISTIC REGRESSION")
print("="*50)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Computed class weights: {class_weight_dict}")

weighted_model = LogisticRegression(
    max_iter=1000, 
    class_weight='balanced', 
    random_state=42
)
weighted_model.fit(X_train, y_train)
weighted_pred = weighted_model.predict(X_test)
weighted_prob = weighted_model.predict_proba(X_test)[:, 1]

print("Weighted Model Results:")
print(f"Accuracy: {accuracy_score(y_test, weighted_pred):.4f}")
print(f"Precision: {precision_score(y_test, weighted_pred):.4f}")
print(f"Recall: {recall_score(y_test, weighted_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, weighted_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, weighted_prob):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, weighted_pred):.4f}")

# ===== MODEL 3: MANUAL OVERSAMPLING + LOGISTIC REGRESSION =====
print("\n" + "="*50)
print("MODEL 3: MANUAL OVERSAMPLING + LOGISTIC REGRESSION")
print("="*50)

def manual_oversample(X, y, random_state=42):
    df_temp = pd.DataFrame(X)
    df_temp['target'] = y
    
    df_majority = df_temp[df_temp.target == 1]
    df_minority = df_temp[df_temp.target == 0]
    
    df_minority_upsampled = resample(
        df_minority, replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    X_resampled = df_upsampled.drop('target', axis=1).values
    y_resampled = df_upsampled['target'].values
    
    return X_resampled, y_resampled

X_train_oversampled, y_train_oversampled = manual_oversample(X_train, y_train)

print(f"After Oversampling - Training set shape: {X_train_oversampled.shape}")
print(f"After Oversampling - Class distribution:")
print(pd.Series(y_train_oversampled).value_counts())

oversampled_model = LogisticRegression(max_iter=1000, random_state=42)
oversampled_model.fit(X_train_oversampled, y_train_oversampled)
oversampled_pred = oversampled_model.predict(X_test)
oversampled_prob = oversampled_model.predict_proba(X_test)[:, 1]

print("Oversampled Model Results:")
print(f"Accuracy: {accuracy_score(y_test, oversampled_pred):.4f}")
print(f"Precision: {precision_score(y_test, oversampled_pred):.4f}")
print(f"Recall: {recall_score(y_test, oversampled_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, oversampled_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, oversampled_prob):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, oversampled_pred):.4f}")

# ===== COMPREHENSIVE EVALUATION =====
print("\n" + "="*50)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*50)

models = {
    'Baseline LR': (baseline_pred, baseline_prob),
    'Weighted LR': (weighted_pred, weighted_prob),
    'Oversampled LR': (oversampled_pred, oversampled_prob)
}

results_df = []
for name, (pred, prob) in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    
    results_df.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'MCC': matthews_corrcoef(y_test, pred),
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    })

results_df = pd.DataFrame(results_df)
print(results_df.round(4))

# ===== VISUALIZATIONS =====

# 1. Model Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].tick_params(axis='x', rotation=45)

axes[0,1].bar(results_df['Model'], results_df['F1'], color='lightgreen')
axes[0,1].set_title('F1 Score Comparison')
axes[0,1].set_ylabel('F1 Score')
axes[0,1].tick_params(axis='x', rotation=45)

axes[1,0].bar(results_df['Model'], results_df['MCC'], color='orange')
axes[1,0].set_title('Matthews Correlation Coefficient')
axes[1,0].set_ylabel('MCC')
axes[1,0].tick_params(axis='x', rotation=45)

axes[1,1].bar(results_df['Model'], results_df['ROC-AUC'], color='coral')
axes[1,1].set_title('ROC-AUC Comparison')
axes[1,1].set_ylabel('ROC-AUC')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']

for i, (name, (pred, prob)) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Logistic Regression Model Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 3. Precision-Recall Curves
plt.figure(figsize=(10, 8))
for i, (name, (pred, prob)) in enumerate(models.items()):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    plt.plot(recall_curve, precision_curve, color=colors[i], lw=2, label=f'{name} (AP = {pr_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Logistic Regression Model Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 4. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (name, (pred, prob)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 5. Logistic Regression Coefficients Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
feature_names = features

models_for_coef = {
    'Baseline LR': baseline_model,
    'Weighted LR': weighted_model,
    'Oversampled LR': oversampled_model
}

for i, (name, model) in enumerate(models_for_coef.items()):
    coef = model.coef_[0]
    feature_importance = pd.Series(coef, index=feature_names)
    feature_importance = feature_importance.reindex(feature_importance.abs().sort_values(ascending=False).index)
    
    feature_importance.plot(kind='bar', color='skyblue', ax=axes[i])
    axes[i].set_title(f"Feature Coefficients - {name}")
    axes[i].set_ylabel("Coefficient Value")
    axes[i].set_xlabel("Features")
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y')

plt.tight_layout()
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

# 7. Top 15 Countries by Cumulative Deaths
plt.figure(figsize=(12, 8))
country_deaths = df.groupby('country')['cumulative_total_deaths'].max().sort_values(ascending=False).head(15)
sns.barplot(x=country_deaths.values, y=country_deaths.index, palette='Reds_r', dodge=False)
plt.title("Top 15 Countries by Cumulative COVID-19 Deaths")
plt.xlabel("Cumulative Deaths")
plt.ylabel("Country")
plt.tight_layout()
plt.show()
