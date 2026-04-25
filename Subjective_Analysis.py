# Combination of all the codes looking into the subjective dataset
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc)

# Calling dataset
data = pd.read_csv("questionnaire_responses.csv")

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

# Converting time played into minutes only And Cleaning Progression
data['time_played'] = data['time_played'].apply(convert_time_to_minutes)
data['progression_clean'] = data['progression'].apply(clean_progression)

# Defining Retention
play_again_map = {"Yes": 1, "No": 0, "Maybe": 0}
data['retention'] = data['Would you play this game again?'].map(play_again_map)


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

# To how retention is split in the dataset
print("\nRetention distribution:\n", y.value_counts())

'''Training'''
# Train Test Split of 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

# The models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

models_tuned = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, C=1, solver='liblinear', random_state=42))
            ]),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
            }

dummy_model = DummyClassifier(strategy='most_frequent')

# Combinding models
all_models = {'Dummy (Baseline)': dummy_model}

for name, model in models.items():
    all_models[f'{name} (Default)'] = model

for name, model in models_tuned.items():
    all_models[f'{name} (Tuned)'] = model

# Training and Evaluating
# To store the data for the AUC graphs
model_results = {}
roc_curves={}
pr_curves={}

# Cross Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in all_models.items():
    # So it only focuses on tuned models
    is_tuned = 'Tuned' in name
    
    # Cross-validation scores
    cv_acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc') if hasattr(model, "predict_proba") else [0]
    
    # Train on full training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Calculate ROC-AUC and PR-AUC
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
    else:
        y_probs = None
        fpr, tpr = [0, 1], [0, 1]
        precision, recall = [1, 0], [0, 1]
        roc_auc = 0.5
        pr_auc = 0.5
    
    # Store results
    model_results[name] = {
        "model": model,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        "cv_acc_mean": cv_acc_scores.mean(),
        "cv_acc_std": cv_acc_scores.std(),
        "cv_roc_mean": cv_roc_scores.mean() if hasattr(model, "predict_proba") else 0.5,
        "cv_roc_std": cv_roc_scores.std() if hasattr(model, "predict_proba") else 0
    }
    
    # To focus only on Tuned Models
    if is_tuned or name == 'Dummy (Baseline)':
        print(f"\n--- {name} ---")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  PR-AUC: {pr_auc:.3f}")
        print(f"  CV Accuracy: {cv_acc_scores.mean():.3f} (±{cv_acc_scores.std():.3f})")

    # Classification report Tuned
    if is_tuned or name == 'Dummy (Baseline)':
        print(f"\nClassification Report - {name}:")
        print(classification_report(y_test, y_pred, 
                target_names=['Low Retention', 'High Retention'], zero_division=0))

    # Confusion Matrix for Tuned models
    if is_tuned or name == 'Dummy (Baseline)':
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap='Blues', 
            ax=ax, display_labels=['Low Retention', 'High Retention']
        )
        ax.set_title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        plt.show()

    # ROC + PR curves
    if hasattr(model, "predict_proba"): #name != "Dummy (Baseline)" and
        y_probs = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_curves[name] = (fpr, tpr, roc_auc)

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_curves[name] = (recall, precision, pr_auc)

    # Feature Importance for tuned
    if is_tuned:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=features)
            importance = importance.sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            fig, ax = plt.subplots(figsize=(8, 5))
            importance.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Feature Importance — {name}')
            ax.set_xlabel('Importance')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.show()
        elif 'Logistic Regression' in name:
            if hasattr(model, 'named_steps'):
                coef = model.named_steps['model'].coef_[0]
            else:
                coef = model.coef_[0]
            importance = pd.Series(coef, index=features).sort_values(ascending=False)

            print("\nFeature Importance:")
            print(importance)

            fig, ax = plt.subplots(figsize=(8, 5))
            importance.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Feature Importance (Coefficients) — {name}')
            ax.set_xlabel('Coefficient Value')
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.show()

# Combined ROC and PR Curves
#  Defining the types of models
default_models = {k: v for k, v in model_results.items() if "(Default)" in k or "Dummy" in k}
tuned_models = {k: v for k, v in model_results.items() if "(Tuned)" in k}

# ROC AUC
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Default on the left
ax = axes[0]
for name, res in default_models.items():
    style = ':' if "Dummy" in name else '--'
    ax.plot(res['fpr'], res['tpr'],
            linestyle=style,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_title("ROC Curve — Default Models")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Tuned on the right
ax = axes[1]
for name, res in tuned_models.items():
    ax.plot(res['fpr'], res['tpr'],
            linestyle='-',
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_title("ROC Curve — Tuned Models")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# PR AUC
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Default on the left
ax = axes[0]
for name, res in default_models.items():
    style = ':' if "Dummy" in name else '--'
    ax.plot(res['recall'], res['precision'],
            linestyle=style,
            label=f"{name} (AUC={res['pr_auc']:.3f})")
ax.set_title("PR Curve — Default Models")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Tuned on the right
ax = axes[1]
for name, res in tuned_models.items():
    ax.plot(res['recall'], res['precision'],
            linestyle='-',
            label=f"{name} (AUC={res['pr_auc']:.3f})")
ax.set_title("PR Curve — Tuned Models")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Finding the best model
best_by_cv = max(model_results, key=lambda x: model_results[x]['cv_acc_mean'])
best_by_roc = max(model_results, key=lambda x: model_results[x]['roc_auc'])

# Best Model Selection
print("\n---BEST MODEL SELECTED---")
print(f"Best model by CV accuracy: {best_by_cv}")
print(f"Best model by ROC-AUC: {best_by_roc}")

# Comparision Table
results_df = pd.DataFrame([
    {
        'Model': name.split(' (')[0],
        'Type': name.split(' (')[1].replace(')', '') if '(' in name else 'Baseline',
        'Test Acc': round(res['test_accuracy'], 3),
        'CV Acc Mean': round(res['cv_acc_mean'], 3),
        'CV Acc Std': round(res['cv_acc_std'], 3),
        'ROC-AUC': round(res['roc_auc'], 3),
        'PR-AUC': round(res['pr_auc'], 3)
    }
    for name, res in model_results.items()
])

# Sort nicely
results_df = results_df.sort_values(by=['Model', 'Type'])

print("\n--- MODEL COMPARISON TABLE ---")
print(results_df.to_string(index=False))

# Showing the differences
comparison_rows = []
for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
    default = model_results[f"{model_name} (Default)"]
    tuned = model_results[f"{model_name} (Tuned)"]
    
    acc_change = tuned["test_accuracy"] - default["test_accuracy"]
    roc_change = tuned["roc_auc"] - default["roc_auc"]
    pr_change = tuned["pr_auc"] - default["pr_auc"]
    
    comparison_rows.append({
        "Model": model_name,
        "Accuracy Change": acc_change,
        "ROC-AUC Change": roc_change,
        "PR-AUC Change": pr_change
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n--- PERFORMANCE COMPARISION ---")
print(comparison_df)

# Features to correlate retention to
corr_cols = ['difficulty', 'difficulty_curve', 'frustration', 'enjoyment', 'played_before', 'retention']

# Overall correlation
print(f'\n{data[corr_cols].corr()}')

# Also print correlations with retention (most relevant)
print("\nCorrelations with retention (sorted):")
corr_with_retention = data[corr_cols].corr()['retention'].sort_values(ascending=False)
for var, corr in corr_with_retention.items():
    print(f"  {var}: {corr:.3f}")

# Bar chart of correlations with retention
fig, ax = plt.subplots(figsize=(8, 5))
corr_with_retention.drop('retention').plot(kind='bar', ax=ax, color='steelblue', edgecolor='black',rot=0)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Features')
ax.set_ylabel('Correlation with Retention')
ax.set_title('Feature Correlations with Retention')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Avg retention per difficulty level
avg_retention = data.groupby('difficulty_curve')['retention'].mean()

# Average retention graph
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(avg_retention.index.astype(str), avg_retention.values,
              color='steelblue', edgecolor='black')
ax.set_xlabel('Perceived Difficulty Increase (1=No increase, 5=Very noticeable)', fontsize=12)
ax.set_ylabel('Retention Rate', fontsize=12)
ax.set_title('Figure: Difficulty Curve vs Player Retention (Subjective Data)', fontsize=14)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()