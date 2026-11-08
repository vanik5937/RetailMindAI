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