import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                             auc, confusion_matrix, classification_report)

# Dataset (old)
df = pd.read_csv(r"C:\Users\kirkl\OneDrive - University of Greenwich\CWs\COMP1682-FYP\Code for FYP\FYP Response.csv")

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

le = LabelEncoder()
for col in ['difficulty', 'difficulty_curve', 'frustration', 'enjoyment']:
    df[col] = le.fit_transform(df[col])

# Cleaning progression
def clean_progression(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    # Removing useless responses
    if "dont remember" in text or "don’t remember" in text:
        return None
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

    return None

# Changing into minutes
def convert_time_to_minutes(time_value):
    if pd.isna(time_value):
        return None
    
    time_value = str(time_value).lower()
    
    try:
        if "hour" in time_value:
            return float(time_value.split()[0]) * 60
        elif "min" in time_value:
            return float(time_value.split()[0])
        elif "year" in time_value:
            return None  # invalid → remove
        else:
            return float(time_value)  # assume already minutes
    except:
        return None

# Converting time played into minutes only
df['time_played'] = df['time_played'].apply(convert_time_to_minutes)
df = df.dropna(subset=['time_played'])

# Apply cleaning
df['progression_clean'] = df['progression'].apply(clean_progression)

# Drop only truly unusable rows
df = df.dropna(subset=['progression_clean'])

# Making target variable
# # median as threshold
threshold = df['progression_clean'].median()
# And Statement
df['retention'] = ((df['progression_clean'] >= threshold) & 
                   (df['play_frequency'] >= 3)).astype(int)
# Indiviudually
# df['retention'] = df['progression_clean'].apply(lambda x:1 if x >= threshold else 0)
# df['retention'] = (df['play_frequency'] >= 3).astype(int)

# Remove any missing data
df = df.dropna()

# Features and target
features = [
    'difficulty',
    'difficulty_curve',
    'frustration',
    'enjoyment',
    # 'time_played',
    # 'play_frequency',
    'played_before'
]

X = df[features]
y = df['retention']

# Training
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Train model
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross Validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print('Cross Validation accuracy:', scores.mean())

# Classification report
print(classification_report(y_test, y_pred))

# Analysing relationship
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

# correlation
print(df[['difficulty', 'retention']].corr())

# Avg retention per difficulty level
avg_retention = df.groupby('difficulty')['retention'].mean()

avg_retention.plot(kind='bar')
plt.xlabel("Difficulty Level")
plt.ylabel("Retention Rate")
plt.title("Difficulty vs Retention")
plt.show()
