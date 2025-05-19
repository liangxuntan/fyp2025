from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import Dict, List

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
    """
    Train and evaluate a multi-label RandomForestClassifier on pfam and media data.

    Parameters
    ----------
    pfam_df : pd.DataFrame
        Pfam counts dataframe (rows=genomes/taxids, columns=Pfam features)
    media_df : pd.DataFrame
        Media traits dataframe (rows=genomes/taxids matching pfam_df index, columns=binary traits)
    test_size : float, optional
        Fraction of data for test set. Default 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default 42.
    n_estimators : int, optional
        Number of trees in RandomForest. Default 500.
    max_depth : int or None, optional
        Max depth of trees. Default None (unlimited).
    stratify : bool, optional
        Whether to stratify split on multi-label targets (default True).
        If True, will use stratification on the most common label if possible.
    save_outputs : bool, optional
        Whether to save train/test splits and predictions as CSV files (default False).
    output_dir : str or Path, optional
        Directory to save output CSV files. Required if save_outputs=True.

    Returns
    -------
    clf : MultiOutputClassifier
        Trained multi-output RandomForestClassifier.
    X_train, X_test, y_train, y_test : pd.DataFrame
        Training and test splits of features and labels.
    y_pred : np.ndarray
        Predictions on X_test.
    """

    # Join data on common index, inner join ensures alignment
    X = pfam_df.join(media_df, how="inner", lsuffix="_pfam")
    y = X[media_df.columns]      # media traits as target
    X = X[pfam_df.columns]       # Pfam features as input

    # Determine stratify labels if requested and possible
    stratify_labels = None
    if stratify:
        # Try to stratify by the most common single label to avoid errors with multi-label stratify
        label_sums = y.sum(axis=0)
        if label_sums.min() > 1:
            most_common_label = label_sums.idxmax()
            stratify_labels = y[most_common_label]
        else:
            print("Warning: stratify=True requested but label counts too sparse; falling back to no stratification.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels)

    # Define base RandomForest with balanced subsample to handle imbalance
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state)

    clf = MultiOutputClassifier(base_rf)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Per-label classification reports
    for i, label in enumerate(y.columns):
        print(f"\nLabel: {label}")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], digits=3))

    # Overall metrics
    exact_match = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nExact-match accuracy: {exact_match:.3f}")
    print(f"Macro-averaged F1    : {macro_f1:.3f}")

    # Save outputs if requested
    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir must be specified if save_outputs=True")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(output_dir / "X_train.csv")
        X_test.to_csv(output_dir / "X_test.csv")
        y_train.to_csv(output_dir / "y_train.csv")
        y_test.to_csv(output_dir / "y_test.csv")

        # Save predictions as DataFrame with same columns and index as y_test
        y_pred_df = pd.DataFrame(y_pred, columns=y.columns, index=y_test.index)
        y_pred_df.to_csv(output_dir / "y_pred.csv")

    return clf, X_train, X_test, y_train, y_test, y_pred

#example usage:
# clf, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_multioutput_rf(
#     pfam_df,
#     media_df,
#     stratify=True,
#     save_outputs=True,
#     output_dir="./model_outputs"
# )


def top_feature_importances(clf: MultiOutputClassifier,
                            X: pd.DataFrame,
                            y: pd.DataFrame,
                            target_label: str,
                            n: int = 10) -> pd.Series:
    """
    Return the top-n feature importances for one target in a multi-output model.

    Parameters
    ----------
    clf : fitted multi-output estimator (e.g., MultiOutputClassifier or Regressor)
    X   : DataFrame of training features (column names are feature names)
    y   : DataFrame of targets (column names are target labels)
    target_label : column in `y` to inspect (e.g., "requires_biotin")
    n   : number of top features to return (default 10)

    Returns
    -------
    pd.Series
        Index = feature names, values = importance scores, sorted descending.
    """
    # Build a mapping of target → feature-importance array
    importances: Dict[str, List[float]] = {
        label: est.feature_importances_
        for label, est in zip(y.columns, clf.estimators_)
    }

    if target_label not in importances:
        raise ValueError(f"'{target_label}' not found among targets: {list(importances)}")

    return (pd.Series(importances[target_label], index=X.columns)
              .sort_values(ascending=False)
              .head(n))

#example usage:top10_pfams = top_feature_importances(clf, X, y, "requires_biotin", n=10)
