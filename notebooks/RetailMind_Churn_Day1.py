# Run this cell once after opening the notebook.
# It checks the project folders and locates the dataset.

from pathlib import Path
import os
import warnings

# The notebook is expected to live in RetailMind_AI/notebooks/.
# In VS Code, open the RetailMind_AI folder as the workspace.
NOTEBOOK_DIR = Path.cwd()
if NOTEBOOK_DIR.name == "notebooks":
    PROJECT_ROOT = NOTEBOOK_DIR.parent
elif (NOTEBOOK_DIR / "notebooks").exists():
    PROJECT_ROOT = NOTEBOOK_DIR
else:
    # Fallback for environments where the current directory is not obvious.
    PROJECT_ROOT = NOTEBOOK_DIR

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    DATA_DIR / "Telco-Customer-Churn.csv",
    DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)

print("Project root:", PROJECT_ROOT)
print("Dataset folder:", DATA_DIR)
print("Models folder:", MODEL_DIR)
print("Reports folder:", REPORT_DIR)

if DATA_PATH is None:
    print("\nDataset not found yet. Download the Kaggle CSV and place it in data/.")
else:
    print("\nDataset found:", DATA_PATH.name)



import numpy as np
import pandas as pd
import joblib
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    learning_curve,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TARGET = "Churn"

print("Imports completed.")


# Locate the CSV again in case the setup cell was run before the file was copied.
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Place the Kaggle CSV in the project's data/ folder. "
        "Expected Telco-Customer-Churn.csv or WA_Fn-UseC_-Telco-Customer-Churn.csv."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())
print("\nColumn data types:")
display(df.dtypes)
print("\nChurn distribution:")
display(df[TARGET].value_counts(dropna=False))
print("\nChurn proportions:")
display(df[TARGET].value_counts(normalize=True).round(3))


# TotalCharges looks numeric but may contain blank strings in the original CSV.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# CustomerID is an identifier, not a behavioural predictor.
df = df.drop(columns=["customerID"], errors="ignore")

# Encode the target: No = 0, Yes = 1.
df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

if df[TARGET].isna().any():
    raise ValueError("Unexpected target labels. Expected only 'Yes' and 'No'.")

X = df.drop(columns=TARGET)
y = df[TARGET].astype(int)

print("Feature matrix:", X.shape)
print("Target vector:", y.shape)
print("\nMissing values by feature (non-zero only):")
display(X.isna().sum()[X.isna().sum() > 0])



# Keep the same split for all four days so that comparisons are consistent.
# Stratification preserves approximately the same churn proportion in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)
print("\nChurn rate — train:", round(y_train.mean(), 3))
print("Churn rate — test: ", round(y_test.mean(), 3))


def make_preprocessor(input_frame):
    """Create numeric and categorical preprocessing based on input columns."""
    numeric_columns = input_frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = input_frame.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def make_model_pipeline(input_frame, estimator, use_feature_selection=False):
    """Package preprocessing and an estimator into one fitted/persistable object."""
    steps = [("preprocess", make_preprocessor(input_frame))]

    if use_feature_selection:
        # k is set later based on the transformed feature count.
        steps.append(("feature_selection", SelectKBest(score_func=f_classif, k=30)))

    steps.append(("model", estimator))
    return Pipeline(steps)


print("Preprocessing and pipeline helper functions are ready.")


# Logistic Regression is the Day 1 baseline.
# class_weight='balanced' asks the model to account for the unequal class counts.
baseline_pipeline = make_model_pipeline(
    X_train,
    LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
)

baseline_pipeline.fit(X_train, y_train)
print("Day 1 baseline pipeline fitted.")


def evaluate_classifier(model_name, fitted_pipeline, features, target, show_details=True):
    """Calculate consistent metrics. Churn=Yes (1) is the positive class."""
    predictions = fitted_pipeline.predict(features)

    if hasattr(fitted_pipeline, "predict_proba"):
        continuous_scores = fitted_pipeline.predict_proba(features)[:, 1]
    else:
        continuous_scores = fitted_pipeline.decision_function(features)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(target, continuous_scores),
    }

    if show_details:
        display(pd.DataFrame([metrics]).set_index("Model"))
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(target, predictions))
        print("\nClassification report:")
        print(classification_report(
            target,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ))

    return metrics


day1_metrics = evaluate_classifier(
    "Day 1 — Logistic Regression baseline",
    baseline_pipeline,
    X_test,
    y_test,
)


# Save the fitted preprocessing + classifier together.
day1_model_path = MODEL_DIR / "churn_day1_baseline.joblib"
joblib.dump(baseline_pipeline, day1_model_path)

# A small inference helper: new customer records go in; labels and probabilities come out.
def predict_churn(customer_rows, fitted_model=baseline_pipeline):
    """Predict churn label and churn probability for a DataFrame of customer records."""
    churn_labels = fitted_model.predict(customer_rows)
    churn_probabilities = fitted_model.predict_proba(customer_rows)[:, 1]

    return pd.DataFrame({
        "predicted_churn": churn_labels,
        "churn_probability": churn_probabilities,
    }, index=customer_rows.index)


print("Saved:", day1_model_path)
display(predict_churn(X_test.head()))