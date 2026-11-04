from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

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