from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Dict, List


def prepare_data(pfam_df, media_df):
    X = pfam_df.join(media_df, how="inner", lsuffix="_pfam")
    y = X[media_df.columns]
    X = X[pfam_df.columns]
    return X, y


def determine_stratify_labels(y, stratify):
    if not stratify:
        return None

    label_sums = y.sum(axis=0)
    if label_sums.min() > 1:
        most_common_label = label_sums.idxmax()
        return y[most_common_label]
    else:
        print("Warning: stratify=True requested but label counts too sparse; falling back to no stratification.")
        return None


def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state
    )
    clf = MultiOutputClassifier(base_rf)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    for i, label in enumerate(y_test.columns):
        print(f"\nLabel: {label}")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], digits=3))

    exact_match = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nExact-match accuracy: {exact_match:.3f}")
    print(f"Macro-averaged F1    : {macro_f1:.3f}")

    return y_pred


def save_results(output_dir, X_train, X_test, y_train, y_test, y_pred, y_columns):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv")
    X_test.to_csv(output_dir / "X_test.csv")
    y_train.to_csv(output_dir / "y_train.csv")
    y_test.to_csv(output_dir / "y_test.csv")

    y_pred_df = pd.DataFrame(y_pred, columns=y_columns, index=y_test.index)
    y_pred_df.to_csv(output_dir / "y_pred.csv")

def train_and_evaluate_multioutput_rf(
    pfam_df,
    media_df,
    test_size=0.2,
    random_state=42,
    n_estimators=500,
    max_depth=None,
    stratify=True,
    save_outputs=False,
    output_dir=None
):
    X, y = prepare_data(pfam_df, media_df)
    stratify_labels = determine_stratify_labels(y, stratify)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels)

    clf = train_model(X_train, y_train, n_estimators, max_depth, random_state)
    y_pred = evaluate_model(clf, X_test, y_test)

    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir must be specified if save_outputs=True")
        save_results(output_dir, X_train, X_test, y_train, y_test, y_pred, y.columns)

    return clf, X_train, X_test, y_train, y_test, y_pred

def get_top_feature_importances(
    clf,
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_label: str,
    n: int = 10
) -> pd.Series:
    """
    Return the top-n feature importances for one target in a multi-output model.

    Parameters
    ----------
    clf : fitted multi-output estimator (e.g., MultiOutputClassifier or Regressor)
        The trained model containing one estimator per target.
    X : pd.DataFrame
        Training features. Columns must match feature names used during training.
    y : pd.DataFrame
        Target labels. Column names correspond to targets.
    target_label : str
        Column in `y` to inspect (e.g., "requires_biotin").
    n : int, optional
        Number of top features to return (default = 10).

    Returns
    -------
    pd.Series
        Feature importance scores for the specified target, sorted in descending order.
        Index = feature names, values = importance scores.
    """
    # Extract feature importances for each target label
    importances: Dict[str, List[float]] = {
        label: est.feature_importances_
        for label, est in zip(y.columns, clf.estimators_)
    }

    if target_label not in importances:
        raise ValueError(f"'{target_label}' not found among targets: {list(importances)}")

    # Create a Series of feature importances for the target and return top-n
    return (pd.Series(importances[target_label], index=X.columns)
              .sort_values(ascending=False)
              .head(n))