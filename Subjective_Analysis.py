# Combination of all the codes looking into the subjective dataset
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc)

# Calling dataset
data = pd.read_csv(r'C:\Users\kirkl\OneDrive - University of Greenwich\CWs\COMP1682-FYP\Code for FYP\FYP Responses Updated.csv')

# Changing into usable features
def group_likert(x):
    if x <= 2:
        return 'Low'
    elif x == 3:
        return 'Neutral'
    else:
        return 'High'

# Mapping Geometry Dash levels
level_mapping = {
    "Stereo Madness": 1,
    "Back on Track": 2,
    "Polargeist": 3,
    "Dry Out": 4,
    "Base After Base": 5,
    "Can't Let Go": 6,
    "Jumper": 7,
    "Time Machine": 8,
    "Cycles": 9,
    "xStep": 10,
    "Clutterfunk": 11,
    "Theory of Everything": 12
}

# Making into simplier features
data['difficulty'] = data['How difficult did you find the game?']
data['difficulty_curve'] = data['Did you feel the difficulty increase?']
data['frustration'] = data['Did you feel frustrated?']
data['enjoyment'] = data['Was it enjoyable?']
data['time_played'] = data['How long did you play before stopping during the time?']
data['progression'] = data['How far did you get within the slotted time? (Level/Progression)']
data['play_frequency'] = data['How often do you play?']
data['played_before'] = data['Have you played this game before?']

# Encoding 
yes_no_map = {"Yes": 1, "No": 0}
data['played_before'] = data['played_before'].map(yes_no_map)

# Cleaning progression
def clean_progression(value):
    if pd.isna(value):
        return 1
    text = str(value).strip().lower()
    # Removing useless responses
    if "dont remember" in text or "don’t remember" in text:
        return 1
    # Level name match
    for level in level_mapping:
        if level.lower() in text:
            return level_mapping[level]
    # "first level"
    if "first level" in text:
        return 1
    # "completed first 2 levels"
    match = re.search(r'(\d+)\s*levels?', text)
    if match:
        return int(match.group(1))
    # "Level 5", "Level 7 20%"
    match = re.search(r'level\s*(\d+)', text)
    if match:
        return int(match.group(1))
    #  "1level 10%"
    match = re.search(r'(\d+)\s*level', text)
    if match:
        return int(match.group(1))
    # Just a number ("6", "10")
    if text.isdigit():
        return int(text)

    return 1

# Minutes converter
def convert_time_to_minutes(time_value):
    if pd.isna(time_value):
        return None
    
    text = str(time_value).lower().strip()

    try:
        if "hour" in text or "hr" in text:
            num = re.search(r'\d+', text)
            return float(num.group()) * 60 if num else None
        elif "min" in text:
            nums = re.findall(r'\d+', text)
            if len(nums) == 2:  # range like 20-25
                return (float(nums[0]) + float(nums[1])) / 2
            elif len(nums) == 1:
                return float(nums[0])
        elif "+" in text:
            num = re.search(r'\d+', text)
            return float(num.group()) if num else None
        elif "year" in text or "month" in text:
            return None  # still invalid
        else:
            return float(text)

    except:
        return None

# Converting time played into minutes only
data['time_played'] = data['time_played'].apply(convert_time_to_minutes)

# Apply cleaning
data['progression_clean'] = data['progression'].apply(clean_progression)

# Making median as threshold
threshold = data['progression_clean'].median()

# Defining Retention
data['retention'] = ((data['progression_clean'] >= threshold) & 
                     (data['play_frequency'] >= 3)).astype(int)


# Features and target
features = [
    'difficulty',
    'difficulty_curve',
    'frustration',
    'enjoyment',
    'played_before'
]

# Defining variables
X = data[features]
y = data['retention']

# To ensure that all data is being used
print("Dataset size:", len(data))
print("\nRetention distribution:\n", y.value_counts())

'''Training'''
# Train Test Split of 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

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

# Overall correlation
print(f'\n{data[['difficulty', 'frustration', 'enjoyment', 'difficulty_curve', 'retention']].corr()}')

# Main Correlation
print(f'\n{data[['difficulty_curve', 'retention']].corr()}')

# Avg retention per difficulty level
avg_retention = data.groupby('difficulty_curve')['retention'].mean()

# Confusion Matrix of Best Model
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title(f"Confusion Matrix ({best_model_name})")
plt.show()

# Average retention graph
avg_retention.plot(kind='bar')
plt.xlabel("Difficulty Curve Level")
plt.ylabel("Retention Rate")
plt.title("Difficulty Curve vs Player Retention")
plt.show()