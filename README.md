# RetailMind AI — Churn Prediction

RetailMind AI is an evolving retail analytics platform. This repository currently contains the first churn-prediction capability, developed incrementally across Days 1–4.

## Project layout

```text
RetailMind_AI/
├── data/                 # Put the Kaggle CSV here
├── notebooks/
│   └── RetailMind_Churn_Days_1_to_4.ipynb
├── models/               # Saved fitted pipelines
├── reports/              # Metrics and learning-curve outputs
├── src/                  # Future reusable application/service code
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

Download Telco Customer Churn from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV in `data/` using either:
- `Telco-Customer-Churn.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Run in VS Code

1. Open this repository folder in VS Code.
2. Install the Python and Jupyter extensions.
3. Create and activate a virtual environment if desired.
4. Install packages: `python -m pip install -r requirements.txt`
5. Open `notebooks/RetailMind_Churn_Days_1_to_4.ipynb`.
6. Select the matching Python kernel.
7. Run the notebook cells from top to bottom.

## Four increments

- Day 1: baseline pipeline, evaluation, persistence, and a prediction helper.
- Day 2: Decision Tree, Random Forest, and SVM comparison.
- Day 3: GridSearchCV, cross-validation, and learning curves.
- Day 4: domain feature engineering, SelectKBest, and ensemble comparison.

The positive target class is `Churn = Yes`. The model-selection metric is mean cross-validation recall. The held-out test set is reserved for final evaluation.

## Notes

XGBoost is optional. If it is not installed, the Day 4 Random Forest experiment still runs.

Do not commit the raw dataset unless its licensing and your repository policy allow it. Review model-artifact size before tracking generated `.joblib` files in Git.
