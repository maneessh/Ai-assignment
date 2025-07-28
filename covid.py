import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("covid_data.csv")

# Create binary target: 1 if daily_new_cases > 0 else 0
df['HasCases'] = df['daily_new_cases'].apply(lambda x: 1 if x > 0 else 0)

# Drop missing values
df = df.dropna()

# Encode categorical 'country'
le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])

# Features and target
features = ['country_encoded', 'cumulative_total_cases', 'active_cases', 'cumulative_total_deaths', 'daily_new_deaths']
X = df[features]
y = df['HasCases']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# Feature Importance (Coefficients)
coef = model.coef_[0]
feature_importance = pd.Series(coef, index=features)
feature_importance = feature_importance.reindex(feature_importance.abs().sort_values(ascending=False).index)

plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar', color='skyblue')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.ylabel("Coefficient value")
plt.xlabel("Features")
plt.grid(axis='y')
plt.show()

# Plot Top 15 Countries by Cumulative Deaths
plt.figure(figsize=(12,6))
country_deaths = df.groupby('country')['cumulative_total_deaths'].max().sort_values(ascending=False).head(15)
sns.barplot(x=country_deaths.values, y=country_deaths.index, palette='Reds_r')
plt.title("Top 15 Countries by Cumulative COVID-19 Deaths")
plt.xlabel("Cumulative Deaths")
plt.ylabel("Country")
plt.show()

# Deaths by Country and Sex (if 'sex' column exists)
if 'sex' in df.columns:
    plt.figure(figsize=(12,6))
    grouped = df.groupby(['country', 'sex'])['daily_new_deaths'].sum().unstack().fillna(0)

    # Filter top 10 countries by total deaths
    top_countries = grouped.sum(axis=1).sort_values(ascending=False).head(10).index
    grouped = grouped.loc[top_countries]

    # Plot
    grouped.plot(kind='bar', stacked=False, figsize=(12,6), colormap='Set2')
    plt.title("COVID-19 Deaths by Country and Sex")
    plt.xlabel("Country")
    plt.ylabel("Total Daily Deaths")
    plt.xticks(rotation=45)
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'sex' not found in dataset — skipping gender-wise death plot.")
