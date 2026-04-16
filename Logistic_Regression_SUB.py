import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, roc_auc_score, precision_recall_curve, auc)

# Calling dataset
df = pd.read_csv(r'C:\Users\kirkl\OneDrive - University of Greenwich\CWs\COMP1682-FYP\Code for FYP\FYP Responses Updated.csv')

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
df['difficulty'] = df['How difficult did you find the game?']
df['difficulty_curve'] = df['Did you feel the difficulty increase?']
df['frustration'] = df['Did you feel frustrated?']
df['enjoyment'] = df['Was it enjoyable?']
df['time_played'] = df['How long did you play before stopping during the time?']
df['progression'] = df['How far did you get within the slotted time? (Level/Progression)']
df['play_frequency'] = df['How often do you play?']
df['played_before'] = df['Have you played this game before?']

# Encoding 
yes_no_map = {"Yes": 1, "No": 0}
df['played_before'] = df['played_before'].map(yes_no_map)

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
df['time_played'] = df['time_played'].apply(convert_time_to_minutes)

# Apply cleaning
df['progression_clean'] = df['progression'].apply(clean_progression)

# median as threshold
threshold = df['progression_clean'].median()

# Defining Retention
df['retention'] = ((df['progression_clean'] >= threshold) & 
                   (df['play_frequency'] >= 3)).astype(int)


# Features and target
features = [
    'difficulty',
    'difficulty_curve',
    'frustration',
    'enjoyment',
    'played_before'
]

# Defining variables
X = df[features]
y = df['retention']

# To ensure that all data is being used
print("Dataset size:", len(df))
print("\nRetention distribution:")
print(df['retention'].value_counts())

'''Training'''
# Standard Split data using the 70-30 rule
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print('---Logistic Regression---')
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# basic cross validation using StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print('Cross Validation accuracy:', scores.mean())

# Classification report
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

# Analysing relationship
if hasattr(model, "feature_importance_"):
    importance = pd.Series(model.feature_importances_, index=X.columns)
elif hasattr(model, "coef_"):
    importance = pd.Series(model.coef_[0], index=X.columns)

print(f'\nFeature Importance: \n{importance.sort_values(ascending=False)}')

# Overall correlation
print(f'\n{df[['difficulty', 'frustration', 'enjoyment', 'difficulty_curve', 'retention']].corr()}')

# Main Correlation
print(f'\n{df[['difficulty_curve', 'retention']].corr()}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

# Avg retention per difficulty level
avg_retention = df.groupby('difficulty_curve')['retention'].mean()

# Average retention graph
avg_retention.plot(kind='bar')
plt.xlabel("Difficulty Curve Level")
plt.ylabel("Retention Rate")
plt.title("Difficulty Curve vs Player Retention")
plt.show()