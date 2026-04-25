# A combination of all the code looking into the objective dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, auc)

# Making the dataset for the code
data = pd.read_csv("games-features.csv")

'''Preprocessing'''
# Showing what the data set changed to 
print("Normal dataset size:", len(data))

# Filtering Platformer Proxy
data = data[
    (data['GenreIsAction'] == 1) &
    (data['GenreIsAdventure'] == 1) &
    (data['GenreIsIndie'] == 1) &
    (data['CategorySinglePlayer'] == 1)
]

# Showing what the data set changed to after makin platformer proxy
print("\nFiltered dataset size for platformer proxy:", len(data))

# Creating Retention
data['retention'] = ((data['SteamSpyPlayersEstimate'] > data['SteamSpyPlayersEstimate'].median())).astype(int)

# Creating difficulty curve
data['AchievementRatio'] = data['AchievementCount'] / (data['AchievementHighlightedCount'] + 1)
data['DifficultyCurve'] = np.log1p(data['AchievementRatio'])

# Features
Features = [
    'DifficultyCurve',
    'AchievementRatio',
    'DLCCount'
]

# Dropping missing values
data = data.dropna(subset=Features + ['retention'])

# Targets
X = data[Features]
y = data['retention']

# To how retention is split in the dataset
print("\nRetention distribution:\n", y.value_counts())

'''Training'''
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

# The models
# Default
models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier()
            }

# Tuned
models_tuned = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000,))
            ]),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
            }

param_grids = {
    'Logistic Regression (Tuned)': {
        'model__C': [0.01, 0.1, 1, 10]
    },
    'Random Forest (Tuned)': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10]
    },
    'Decision Tree (Tuned)': {
        'max_depth': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4]
    }
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
    # To focus on Tuned 
    is_tuned = '(Tuned)' in name

    if is_tuned and name in param_grids:
        grid = GridSearchCV(
            model,
            param_grids[name],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_  
        print(f"\nBest params for {name}: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc') if hasattr(model, "predict_proba") else [0]
    
    # Train on full training set
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
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "cv_acc_mean": cv_acc_scores.mean(),
        "cv_acc_std": cv_acc_scores.std(),
        "cv_roc_mean": cv_roc_scores.mean() if hasattr(model, "predict_proba") else 0.5,
        "cv_roc_std": cv_roc_scores.std() if hasattr(model, "predict_proba") else 0
    }

    # For tuned specifically
    if is_tuned or name == 'Dummy (Baseline)':
        print(f"\n--- {name} ---")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  PR-AUC: {pr_auc:.3f}")
        print(f"  CV Accuracy: {cv_acc_scores.mean():.3f} (±{cv_acc_scores.std():.3f})")

    # Classification report for tuned
    if is_tuned or name == 'Dummy (Baseline)':
        print(f"\nClassification Report - {name}:")
        print(classification_report(y_test, y_pred, target_names=['Low Retention', 'High Retention'], zero_division=0))

    # Confusion Matrix for tuned only
    if is_tuned or name == 'Dummy (Baseline)':
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap='Blues', ax=ax,
            display_labels=['Low Retention', 'High Retention']
        )
        ax.set_title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        plt.show()

    # Feature Importance for tuned
    if is_tuned:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=Features)
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
            importance = pd.Series(coef, index=Features).sort_values(ascending=False)

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

#  Finding best model
best_by_cv = max(model_results, key=lambda x: model_results[x]['cv_acc_mean'])
best_by_roc = max(model_results, key=lambda x: model_results[x]['roc_auc'])
#  Best Model Selection
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

# Correlation of all relevant features
corr_cols = ['DifficultyCurve', 'RecommendationCount', 'DLCCount', 'retention']

# Correlation table
print(f'\n{data[corr_cols].corr()}')

# Also print correlations with retention (most relevant)
print("\nCorrelations with retention (sorted):")
corr_with_retention = data[corr_cols].corr()['retention'].sort_values(ascending=False)
for var, corr in corr_with_retention.items():
    print(f"  {var}: {corr:.3f}")

# Bar chart of correlations with retention
fig, ax = plt.subplots(figsize=(8, 5))
corr_with_retention.drop('retention').plot(kind='bar', ax=ax, color='steelblue', edgecolor='black',rot=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Features')
ax.set_ylabel('Correlation with Retention')
ax.set_title('Feature Correlations with Retention')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Create Difficulty Curve levels (bins since it is numerical only)
data['DifficultyCurveLevel'] = pd.qcut(data['DifficultyCurve'], q=4, labels=['Low', 'Medium', 'High'], duplicates='drop')

# Calculate average retention per level
avg_retention = data.groupby('DifficultyCurveLevel', observed=True)['retention'].mean()

# Average retention graph
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(avg_retention.index.astype(str), avg_retention.values,
              color='steelblue', edgecolor='black')
ax.set_xlabel('Perceived Difficulty Curve', fontsize=12)
ax.set_ylabel('Retention Rate', fontsize=12)
ax.set_title('Figure: Difficulty Curve vs Player Retention (Binned Objective Data)', fontsize=14)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()