# A combination of all the code looking into the objective dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc)

# Making the dataset for the code
data = pd.read_csv(r'C:\Users\kirkl\OneDrive - University of Greenwich\CWs\COMP1682-FYP\Code for FYP\games-features.csv')

'''Preprocessing'''
# Showing what the data set changed to 
print("Normal dataset size:", len(data))

# Filtering Platformer Proxy
data = data[
    (data['GenreIsAction'] == 1) &
    (data['CategorySinglePlayer'] == 1) &
    (data['GenreIsIndie'] == 1)
]

# Showing what the data set changed to 
print("\nFiltered dataset size:", len(data))

# Creating Retention
data['retention'] = (data['SteamSpyPlayersEstimate'] > data['SteamSpyPlayersEstimate'].median()).astype(int)

# Creating difficulty curve
data['DifficultyCurve'] = (data['AchievementCount'] / (data['AchievementHighlightedCount'] + 1)) * np.log1p(data['SteamSpyPlayersEstimate'])

# Features
Features = [
    'DifficultyCurve',
    'AchievementCount',
    'AchievementHighlightedCount',
    'RecommendationCount',
    'DLCCount'
]

# Dropping missing values
data = data.dropna(subset=Features + ['retention'])

# Targets
X = data[Features]
y = data['retention']

'''Training'''
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# The models
models = {
            'Dummy (Baseline)': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=42)
            }

# Training and Evaluating
model_results = {}

# Cross Validation Setup
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    # Cross Validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Training split
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_probs)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
    else:
        roc, pr_auc = 0, 0

    model_results[name] = {
        "model": model,
        "accuracy": acc,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "cv_mean": cv_mean,
        "cv_std": cv_std
    }

    print(f"\n---{name}---")
    print(f"Accuracy: {acc:.3f}")
    print(f"Cross Validation Mean Accuracy: {cv_mean:.3f}")
    print(f"Cross Validation Standard Deviation: {cv_std:.3f}")
    print(f"ROC AUC: {roc:.3f} and PR AUC: {pr_auc:.3f}")

# Best Model Selection
# Using Cross Validation mean accuracy
best_model_name = max(model_results, key=lambda x: model_results[x]['cv_mean'])
best_model = model_results[best_model_name]["model"]
print(f"\nBest Model (based on CV): {best_model_name}")

# Using roc_auc
best_model_name = max(model_results, key=lambda x: model_results[x]['roc_auc'])
best_model = model_results[best_model_name]["model"]
print(f"\nBest Model (for ROC AUC): {best_model_name}")

# Classification report for the best model
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]
print(f'\n{classification_report(y_test, y_pred)}')

# Visualising ROC and PR AUC
roc = roc_auc_score(y_test, y_probs)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# ROC AUC
axes[0].plot(fpr, tpr, label=f"AUC = {roc:.2f}")
axes[0].plot([0, 1], [0, 1], linestyle='--')
axes[0].set_title(f"ROC Curve ({best_model_name})")
axes[0].legend()

# PR AUC
axes[1].plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
axes[1].set_title(f"PR Curve ({best_model_name})")
axes[1].legend()

plt.tight_layout()
plt.show()

# Analysing relationship
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nFeature Importance:\n", importance.sort_values(ascending=False))

# Correlation of all relevant features
print(f'\n{data[['DifficultyCurve', 'AchievementCount', 'AchievementHighlightedCount', 'RecommendationCount', 'DLCCount', 'retention']].corr()}')

# Correlation between difficulty and retention
print(f'\n{data[['DifficultyCurve', 'retention']].corr()}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title(f"Confusion Matrix ({best_model_name})")
plt.show()

# Create Difficulty Curve levels (bins since it is numerical only)
data['DifficultyCurveLevel'] = pd.qcut(data['DifficultyCurve'], q=4, labels=['Low', 'Medium', 'High'], duplicates='drop')

# Calculate average retention per level
avg_retention_curve = data.groupby('DifficultyCurveLevel', observed=True)['retention'].mean()

# Plot bar chart
avg_retention_curve.plot(kind='bar', rot=0)
plt.xlabel("Difficulty Curve Level")
plt.ylabel("Retention Rate")
plt.title("Difficulty Curve vs Player Retention")
plt.show()