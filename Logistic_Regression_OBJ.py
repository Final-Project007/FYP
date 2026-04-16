import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, auc,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve)

# Making the dataset for the code
ds = pd.read_csv(r'C:\Users\kirkl\OneDrive - University of Greenwich\CWs\COMP1682-FYP\Code for FYP\games-features.csv')

'''Preprocessing'''
# Filtering Platformer Proxy
ds = ds[
    (ds['GenreIsAction'] == 1) &
    (ds['CategorySinglePlayer'] == 1) &
    (ds['GenreIsIndie'] == 1)
]

# Creating Retention
ds['retention'] = (ds['SteamSpyPlayersEstimate'] > ds['SteamSpyPlayersEstimate'].median()).astype(int)

# Creating difficulty curve
ds['DifficultyCurve'] = (ds['AchievementCount'] / (ds['AchievementHighlightedCount'] + 1)) * np.log1p(ds['SteamSpyPlayersEstimate'])

# Features
Features = [
    'DifficultyCurve',
    'AchievementCount',
    'AchievementHighlightedCount',
    'RecommendationCount',
    'DLCCount'
]

# Variables
X = ds[Features]
y = ds['retention']

# Dropping missing values
ds = ds.dropna(subset=Features + ['retention'])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluating model
y_pred = model.predict(X_test)

# Accuracy Score
print('---Logistic Regression---')
print('Accuracy:', accuracy_score(y_test, y_pred))

# Cross Validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print('Cross Validation accuracy:', scores.mean())
print('Cross Validation std:', scores.std())

# Classification Report
print(f'\n{classification_report(y_test, y_pred)}')

# AUC Scores
# Receiver Operating Characteristic 
y_probs = model.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_probs)
print('ROC AUC:', roc)

# Precision Recall
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
print('PR AUC:', pr_auc)

# Visualising ROC and PR AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# ROC AUC
axes[0].plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.2f})")
axes[0].plot([0, 1], [0, 1], linestyle='--')  # random baseline
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

# PR AUC
axes[1].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.show()

# Showing Feature importance
if hasattr(model, "feature_importance_"):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print(f'\nFeature Importance: \n{importance.sort_values(ascending=False)}')

# Correlation of all relevant features
print(f'\n{ds[['DifficultyCurve', 'AchievementCount', 'AchievementHighlightedCount', 'RecommendationCount', 'DLCCount', 'retention']].corr()}')

# Correlation between difficulty and retention
print(f'\n{ds[['DifficultyCurve', 'retention']].corr()}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

# Create Difficulty Curve levels (bins since it is numerical only)
ds['DifficultyCurveLevel'] = pd.qcut(ds['DifficultyCurve'], q=4, labels=['Low', 'Medium', 'High'], duplicates='drop')

# Calculate average retention per level
avg_retention_curve = ds.groupby('DifficultyCurveLevel', observed=True)['retention'].mean()

# Plot bar chart
avg_retention_curve.plot(kind='bar', rot=0)
plt.xlabel("Difficulty Curve Level")
plt.ylabel("Retention Rate")
plt.title("Difficulty Curve vs Player Retention")
plt.show()