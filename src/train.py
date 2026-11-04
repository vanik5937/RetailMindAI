# Logistic Regression is the Day 1 baseline.
# class_weight='balanced' asks the model to account for the unequal class counts.
from preprocessing import make_model_pipeline
from sklearn.linear_model  import LogisticRegression

baseline_pipeline = make_model_pipeline(
    X_train,
    LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
)

baseline_pipeline.fit(X_train, y_train)