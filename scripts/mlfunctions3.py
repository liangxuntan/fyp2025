import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV,KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

def read_csv(filepath):
    """
    Reads a CSV file into a pandas DataFrame with the first column as the index.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the first column set as the index.
    """
    return pd.read_csv(filepath, index_col=0)

def read_json(filepath):
    """
    Reads a JSON file as a dictionary and converts all top-level keys to integers.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary with integer keys.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert all keys to int (top-level only)
    return {str(k): v for k, v in data.items()}

def check_and_align_indices(df1, df2, drop_diff=True, verbose=True):
    """
    Ensure two DataFrames have exactly matching indices (order and values).
    Optionally drops non-shared rows.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        The two DataFrames to align.
    drop_diff : bool, default=True
        If True, keep only the intersecting rows (indices) and sort in the same order.
        If False, raises an error if indices do not match.
    verbose : bool, default=True
        Print messages about what was done.
    
    Returns
    -------
    df1_aligned, df2_aligned : pd.DataFrame
        Aligned DataFrames with same index and order.
    """
    idx1 = set(df1.index)
    idx2 = set(df2.index)
    if idx1 == idx2:
        # They have the same indices (but may be in different orders)
        df1a = df1.loc[sorted(df1.index)]
        df2a = df2.loc[sorted(df1.index)]
        if not df1.index.equals(df2.index):
            if verbose:
                print("Indices had same values but different order; both DataFrames re-sorted.")
        return df1a, df2a
    else:
        # Indices don't match exactly
        intersect_idx = sorted(idx1 & idx2)
        missing1 = idx1 - idx2
        missing2 = idx2 - idx1
        if verbose:
            print(f"Warning: DataFrames have non-matching indices.")
            if missing1:
                print(f"  {len(missing1)} indices in df1 not in df2: {list(missing1)[:5]} ...")
            if missing2:
                print(f"  {len(missing2)} indices in df2 not in df1: {list(missing2)[:5]} ...")
            print(f"  Keeping only {len(intersect_idx)} shared indices.")
        if not drop_diff:
            raise ValueError("Indices of input DataFrames do not match!")
        # Subset and re-order both DataFrames
        df1a = df1.loc[intersect_idx]
        df2a = df2.loc[intersect_idx]
        return df1a, df2a

def run_rf_multilabel_classification(
    pfam_df, media_df, test_size=0.2, random_state=42, stratify=None, verbose=True, cv=None
):
    """
    Trains and evaluates a multi-output Random Forest classifier, with optional cross-validation.

    Returns:
    - If cv is None: clf, X_test, y_test, y_pred
    - If cv is not None: dict with mean/std metrics and per-fold metrics list
    """
    pfam_df, media_df = check_and_align_indices(pfam_df, media_df, drop_diff=True, verbose=verbose)
    X, y = pfam_df, media_df

    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=4,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state
    )

    if cv is not None:
        from sklearn.model_selection import KFold
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf = MultiOutputClassifier(base_rf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # Use your metrics collection function
            d = collect_metrics_dict(
                setting=f"CV fold {fold}/{cv}",
                y_true=y_test,
                y_pred=y_pred,
                xtest=X_test,
                n_estimators=base_rf.n_estimators
            )
            results.append(d)
            if verbose:
                print(f"\n=== Fold {fold} ===")
                print("Accuracy:", d["Accuracy"])
                print("F1 Micro:", d["F1 Micro"])
                print("F1 Macro:", d["F1 Macro"])
        # Mean ± std summary
        accs = [float(r["Accuracy"]) for r in results]
        f1_micros = [float(r["F1 Micro"]) for r in results]
        f1_macros = [float(r["F1 Macro"]) for r in results]
        if verbose:
            print("\n=== Cross-Validation Results ===")
            print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"Mean F1 Micro: {np.mean(f1_micros):.4f} ± {np.std(f1_micros):.4f}")
            print(f"Mean F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
        return {
            'cv_accuracy_mean': np.mean(accs), 'cv_accuracy_std': np.std(accs),
            'cv_f1_micro_mean': np.mean(f1_micros), 'cv_f1_micro_std': np.std(f1_micros),
            'cv_f1_macro_mean': np.mean(f1_macros), 'cv_f1_macro_std': np.std(f1_macros),
            'fold_metrics': results,
        }
    else:
        # Normal train/test split
        from sklearn.model_selection import train_test_split
        clf = MultiOutputClassifier(base_rf)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if verbose:
            print("=== Accuracy Score ===")
            print(accuracy_score(y_test, y_pred))
            print("\n=== F1 Micro ===")
            print(f1_score(y_test, y_pred, average="micro"))
            print("\n=== F1 Macro ===")
            print(f1_score(y_test, y_pred, average="macro"))
            print("\n=== Classification Report ===")
            print(classification_report(y_test, y_pred, target_names=list(y.columns)))
        return clf, X_test, y_test, y_pred
    
def run_rf_mlc_trainmodel(
    pfam_df, media_df, random_state=42
):
    """
    Fit a multi-output (multi-label) Random Forest classifier.
    
    Parameters
    ----------
    pfam_df : pd.DataFrame
        Feature matrix (samples x features)
    media_df : pd.DataFrame
        Label matrix (samples x labels)
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    clf : MultiOutputClassifier
        Fitted multi-label random forest classifier
    """
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=4,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state
    )

    clf = MultiOutputClassifier(base_rf)
    clf.fit(pfam_df, media_df)
    return clf

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def run_rf_multilabel_regression(
    pfam_df, media_df, test_size=0.2, random_state=42, verbose=True, cv=None
):
    """
    Trains and evaluates a multi-output Random Forest regressor, with optional cross-validation.

    Parameters
    ----------
    pfam_df : pd.DataFrame
        Features (e.g., Pfam binary/presence matrix or other features).
    media_df : pd.DataFrame
        Non-binary target variables (e.g., ingredient concentrations).
    test_size : float, default=0.2
        Fraction of data used for test set (if cv is None).
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress and metrics.
    cv : int or None, default=None
        If set (e.g., 5), performs KFold cross-validation.

    Returns
    -------
    - If cv is None: reg, X_test, y_test, y_pred
    - If cv is not None: dict with mean/std metrics and per-fold metrics list
    """
    # Align indices if needed
    pfam_df, media_df = pfam_df.align(media_df, join='inner', axis=0)
    X, y = pfam_df, media_df

    base_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=random_state
    )

    if cv is not None:
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            reg = MultiOutputRegressor(base_rf)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
            mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
            mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
            fold_metrics = {
                "fold": fold,
                "R2": r2,
                "MSE": mse,
                "MAE": mae,
            }
            results.append(fold_metrics)
            if verbose:
                print(f"\n=== Fold {fold} ===")
                print("R2 Score:", r2)
                print("MSE:", mse)
                print("MAE:", mae)
        r2s = [r["R2"] for r in results]
        mses = [r["MSE"] for r in results]
        maes = [r["MAE"] for r in results]
        if verbose:
            print("\n=== Cross-Validation Results ===")
            print(f"Mean R2: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"Mean MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
            print(f"Mean MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        return {
            'cv_r2_mean': np.mean(r2s), 'cv_r2_std': np.std(r2s),
            'cv_mse_mean': np.mean(mses), 'cv_mse_std': np.std(mses),
            'cv_mae_mean': np.mean(maes), 'cv_mae_std': np.std(maes),
            'fold_metrics': results,
        }
    else:
        # Normal train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        reg = MultiOutputRegressor(base_rf)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
        mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
        if verbose:
            print("=== Regression Metrics ===")
            print("R2 Score:", r2)
            print("MSE:", mse)
            print("MAE:", mae)
        return reg, X_test, y_test, y_pred


from sklearn.metrics import accuracy_score, f1_score

def collect_metrics_dict(
    setting,
    y_true,
    y_pred,
    xtest,
    n_labels=None,
    n_features=None,
    n_samples=None,
    n_estimators=None,
    best_params=None
):
    """
    Returns a dictionary of performance metrics for easy reporting.
    """
    d = {
        "Setting": setting,
        "n_samples": y_true.shape[0],
        "n_labels": y_true.shape[1],
        "n_features": xtest.shape[1],
        "Accuracy": f"{accuracy_score(y_true, y_pred):.3f}",
        "F1 Micro": f"{f1_score(y_true, y_pred, average='micro'):.3f}",
        "F1 Macro": f"{f1_score(y_true, y_pred, average='macro'):.3f}",
    }
    if n_estimators is not None:
        d["n_estimators"] = n_estimators
    if best_params is not None:
        d["Best Params"] = str(best_params)
    return d


def report_metrics_table(
    results_list,
    columns=None,
    title="Model Performance Summary",
    save_path=None,
    dpi=600,
    fontsize=16
):
    """
    Plot and return a summary table of model performance metrics for a report.

    Parameters
    ----------
    results_list : list of dict
        Each dict is a row with keys as column names, e.g.:
        [
          {'Setting': 'CV, min=5', 'Accuracy': '0.012 ± 0.001', 'F1 Micro': '0.51 ± 0.03', ...},
          {'Setting': 'Test set', ...},
        ]
    columns : list of str, optional
        Specify column order and labels. If None, inferred from first row.
    title : str
        Title for the table figure.
    save_path : str, optional
        Path to save the figure.
    dpi : int
        DPI for saving.
    fontsize : int
        Table font size.
    """
    if columns is None:
        columns = list(results_list[0].keys())
    df = pd.DataFrame(results_list, columns=columns)

    # Plot as table
    plt.figure(figsize=(min(2 + 2*len(columns), 15), 1 + 0.5*len(df)))
    plt.axis('off')
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_fontsize(fontsize+2)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E1EAF2')
        cell.set_linewidth(0.5)
        cell.set_height(0.7)
    plt.title(title, fontsize=fontsize+4, pad=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Table saved to: {save_path}")
    plt.show()
    return df

import textwrap

def report_metrics_table2(
    results_list,
    columns=None,
    title="Model Performance Summary",
    save_path=None,
    dpi=600,
    fontsize=16,
    min_colwidth=32,    # NEW: Minimum width of columns (in characters)
    cell_height=0.36,   # NEW: Height of each cell/row
    params_col='Params'
):
    if columns is None:
        columns = list(results_list[0].keys())
    df = pd.DataFrame(results_list, columns=columns)
    
    # Wrap all columns for multi-line (including headings)
    def wrap(x, width=min_colwidth):
        return "\n".join(textwrap.wrap(str(x), width=width))
    df = df.applymap(lambda x: wrap(x, min_colwidth))
    col_labels = [wrap(col, min_colwidth) for col in df.columns]
    
    n_cols = len(df.columns)
    colWidths = [min_colwidth/15. for _ in df.columns]  # Each column this wide
    table_width = sum(colWidths)
    fig_w = max(table_width, 10)
    fig_h = max(1.1 + cell_height*len(df), 2.7)
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('off')
    table = plt.table(
        cellText=df.values,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1],
        colWidths=[w/table_width for w in colWidths]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize if n_cols < 8 else max(10, fontsize-2))
    # Style header and adjust cell size
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_linewidth(0.5)
        cell.set_height(cell_height)  # <-- Use your option!
        if row == 0:
            cell.set_fontsize(fontsize+2)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E1EAF2')
        if df.columns[col] == params_col:
            cell.set_text_props(ha='left')
    plt.title(title, fontsize=fontsize+4, pad=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Table saved to: {save_path}")
    plt.show()
    return df

def report_label_distribution(label_df, plot=True, top_n=20, axis_fontsize=16):
    """
    Print and optionally plot the distribution of labels in a multi-label DataFrame.

    Parameters
    ----------
    label_df : pd.DataFrame
        DataFrame where rows are samples and columns are labels (0/1).
    plot : bool, default=True
        Whether to show a bar plot of label counts.
    top_n : int, default=20
        Show only the top N most common labels in the plot.
    axis_fontsize : int, default=14
        Font size for axis labels and ticks.
    """
    # Count positives per label
    label_counts = label_df.sum(axis=0).sort_values(ascending=False)
    label_fraction = label_df.mean(axis=0).sort_values(ascending=False)
    
    print("=== Label Distribution Report ===")
    print(f"Total labels (columns): {label_df.shape[1]}")
    print(f"Total samples (rows): {label_df.shape[0]}")
    print("\nLabels with the most positives (top 10):")
    print(label_counts.head(10))
    print("\nFraction of samples positive for each of top 10 labels:")
    print(label_fraction.head(10).apply(lambda x: f"{x:.3f}"))

    print("\nNumber of labels with zero positives:", (label_counts == 0).sum())
    print("Number of labels with only one positive:", (label_counts == 1).sum())
    print("Number of labels present in more than 10% of samples:",
          (label_fraction > 0.1).sum())

    print("\n=== Label Distribution (Full, sorted) ===")
    print(label_counts)

    # Per-sample label count stats
    sample_label_counts = label_df.sum(axis=1)
    print("\nPer-sample label count summary:")
    print(sample_label_counts.describe())

    # Plot (optional, top_n only for clarity)
    if plot:
        plt.figure(figsize=(min(16, top_n * 0.7), 4))
        ax = label_counts.head(top_n).plot(kind='bar')
        plt.title(f"Top {top_n} Most Common Labels", fontsize=axis_fontsize+2)
        plt.xlabel("Label", fontsize=axis_fontsize)
        plt.ylabel("Number of Media", fontsize=axis_fontsize)
        ax.tick_params(axis='x', labelsize=axis_fontsize-2, rotation=60)
        ax.tick_params(axis='y', labelsize=axis_fontsize-2)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

def report_feature_distribution(feature_df, plot=True, top_n=20, axis_fontsize=16):
    """
    Print and optionally plot the distribution of features in a DataFrame.
    """
    import matplotlib.pyplot as plt
    feature_df = feature_df.fillna(0)  # ensure no NaN
    feature_counts = feature_df.sum(axis=0).sort_values(ascending=False)
    feature_fraction = feature_df.mean(axis=0).sort_values(ascending=False)

    print("=== Feature Distribution Report ===")
    print(f"Total features (columns): {feature_df.shape[1]}")
    print(f"Total samples (rows): {feature_df.shape[0]}")
    print("\nFeatures present in the most samples (top 10):")
    print(feature_counts.head(10))
    print("\nFraction of samples positive for each of top 10 features:")
    print(feature_fraction.head(10).apply(lambda x: f'{x:.3f}'))

    print("\nNumber of features present in zero samples:", (feature_counts == 0).sum())
    print("Number of features present in only one sample:", (feature_counts == 1).sum())
    print("Number of features present in more than 10% of samples:",
          (feature_fraction > 0.1).sum())

    print("\n=== Feature Distribution (Full, sorted) ===")
    print(feature_counts)

    # Per-sample feature count stats
    sample_feature_counts = (feature_df != 0).sum(axis=1)
    print("\nPer-sample feature count summary:")
    print(sample_feature_counts.describe())

    if plot:
        plt.figure(figsize=(min(16, top_n * 0.7), 4))
        ax = feature_counts.head(top_n).plot(kind='bar')
        plt.title(f"Top {top_n} Most Common Features", fontsize=axis_fontsize+2)
        plt.xlabel("Feature", fontsize=axis_fontsize)
        plt.ylabel("Number of Species", fontsize=axis_fontsize)
        ax.tick_params(axis='x', labelsize=axis_fontsize-2, rotation=60)
        ax.tick_params(axis='y', labelsize=axis_fontsize-2)
        ax.grid(False)
        ax.set_ylim(0, feature_counts.head(top_n).max() * 1.1)
        plt.tight_layout()
        plt.show()

def plot_feature_distribution(feature_df, top_n=30, figsize=(14, 4), log_scale=False):
    """
    Plot the number of samples each feature (column) is present in.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame with rows as samples and columns as features (Pfam domains).
    top_n : int, default=30
        Show only the top N most common features for clarity.
    figsize : tuple, default=(14,4)
        Size of the figure.
    log_scale : bool, default=False
        If True, use log scale for the y-axis (useful if very skewed).
    """
    feat_counts = (feature_df != 0).sum(axis=0).sort_values(ascending=False)
    plt.figure(figsize=figsize)
    feat_counts.head(top_n).plot(kind='bar')
    plt.xlabel("Feature (Pfam Domain)")
    plt.ylabel("Number of Samples with Feature Present")
    plt.title(f"Top {top_n} Most Common Pfam Features")
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Number of Samples (log scale)")
    plt.tight_layout()
    plt.show()

    print(f"Total features: {len(feat_counts)}")
    print(f"Features present in at least 1 sample: {(feat_counts > 0).sum()}")
    print(f"Features present in ALL samples: {(feat_counts == feature_df.shape[0]).sum()}")
    print(f"Most common feature present in {feat_counts.iloc[0]} samples.")

def plot_feature_presence_hist(feature_df, bins=50, log_scale=True):
    """
    Plot histogram of feature (Pfam) presence across samples.
    """
    feat_counts = (feature_df != 0).sum(axis=0)
    plt.figure(figsize=(8, 4))
    plt.hist(feat_counts, bins=bins, log=log_scale, edgecolor='k')
    plt.xlabel("Number of Samples Feature is Present In")
    plt.ylabel("Number of Features")
    plt.title("Histogram of Feature (Pfam) Presence")
    plt.tight_layout()
    plt.show()

    print(f"Median feature presence: {feat_counts.median()}")
    print(f"Mean feature presence: {feat_counts.mean():.2f}")
    print(f"Number of features present in only one sample: {(feat_counts==1).sum()}")
    print(f"Number of features present in all samples: {(feat_counts==feature_df.shape[0]).sum()}")


def plot_feature_presence_cdf(feature_df):
    feat_counts = (feature_df != 0).sum(axis=0).values
    feat_counts_sorted = np.sort(feat_counts)
    yvals = np.arange(1, len(feat_counts_sorted)+1) / len(feat_counts_sorted)
    plt.figure(figsize=(7,4))
    plt.plot(feat_counts_sorted, yvals)
    plt.xlabel("Samples feature is present in")
    plt.ylabel("Cumulative fraction of features")
    plt.title("CDF: Feature Presence Across Samples")
    plt.tight_layout()
    plt.show()

def summarize_feature_distribution_pub(feature_df, bins=50, figsize=(12,5), log_hist=True, show_cdf=True, save_path=None, dpi=300):
    """
    Publication-quality panel: histogram and CDF of feature (Pfam) coverage.
    Optionally saves as PNG/PDF.
    """
    # Use seaborn-v0_8-whitegrid style for modern matplotlib
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')
        
    feat_counts = (feature_df != 0).sum(axis=0).values
    n_features = len(feat_counts)
    n_samples = feature_df.shape[0]

    fig, axs = plt.subplots(1, 2 if show_cdf else 1, figsize=figsize)

    # Font sizes for publication
    fontsize = 20
    ticksize = 14
    titlesize = 18

    # Histogram
    ax = axs[0] if show_cdf else axs
    ax.hist(feat_counts, bins=bins, log=log_hist, edgecolor='black', linewidth=1.2, color="#4F81BD")
    ax.set_xlabel("Number of species containing feature", fontsize=fontsize)
    ax.set_ylabel("Number of features", fontsize=fontsize)
    ax.set_title("Feature (Pfam) Coverage Histogram", fontsize=titlesize, pad=15)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

    # CDF
    if show_cdf:
        feat_counts_sorted = np.sort(feat_counts)
        yvals = np.arange(1, n_features+1) / n_features
        axs[1].plot(feat_counts_sorted, yvals, color="#C0504D", linewidth=2)
        axs[1].set_xlabel("Number of species containing feature", fontsize=fontsize)
        axs[1].set_ylabel("Cumulative fraction of features", fontsize=fontsize)
        axs[1].set_title("CDF: Feature Coverage", fontsize=titlesize, pad=15)
        axs[1].tick_params(axis='both', labelsize=ticksize)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

    plt.tight_layout()

    # Save as PNG/PDF
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print stats
    print("=== Feature (Pfam) Coverage Summary ===")
    print(f"Total features: {n_features}")
    print(f"Total samples: {n_samples}")
    print(f"Mean feature presence: {feat_counts.mean():.2f}")
    print(f"Median feature presence: {np.median(feat_counts)}")
    print(f"Min/Max feature presence: {feat_counts.min()} / {feat_counts.max()}")
    print(f"Features present in only one sample: {(feat_counts == 1).sum()}")
    print(f"Features present in ALL samples: {(feat_counts == n_samples).sum()}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"{int(q*100)}% of features present in ≤ {np.percentile(feat_counts, q*100):.0f} samples")

def plot_selected_feature_distributions(
    feature_df, features, bins=50, figsize=(6, 4), show_cdf=True,
    save_path=None, dpi=300
):
    """
    Plots histogram and CDF for selected features (columns) from feature_df.
    - No gridlines
    - X-axis uses integer ticks only
    - Linear y-axis (not log)
    """
    from matplotlib.ticker import MaxNLocator

    n_feats = len(features)
    plt.style.use('seaborn-v0_8-whitegrid')
    fontsize = 15
    ticksize = 12

    ncols = 2 if show_cdf else 1
    fig, axs = plt.subplots(n_feats, ncols, figsize=(figsize[0]*ncols, figsize[1]*n_feats), squeeze=False)
    for i, feat in enumerate(features):
        data = feature_df[feat].values
        # Histogram
        ax = axs[i, 0]
        ax.hist(data, bins=bins, log=False, color="#4F81BD", edgecolor='black', linewidth=1.2)
        ax.set_xlabel(f"{feat} value", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.set_title(f"{feat} Histogram", fontsize=fontsize+2)
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # CDF
        if show_cdf:
            ax2 = axs[i, 1]
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(data)+1) / len(data)
            ax2.plot(sorted_data, cdf, color="#C0504D", linewidth=2)
            ax2.set_xlabel(f"{feat} value", fontsize=fontsize)
            ax2.set_ylabel("CDF", fontsize=fontsize)
            ax2.set_title(f"{feat} CDF", fontsize=fontsize+2)
            ax2.tick_params(axis='both', labelsize=ticksize)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(False)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {save_path}")
    plt.show()

from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from collections import Counter, defaultdict

def plot_feature_histogram_by_taxonomy(
    feature_df, features, tax_dict, bar_width=0.7, figsize=(6, 4),
    show_cdf=False, save_path=None, dpi=300, palette='tab20',
    max_taxa=None, count_threshold=None, taxonomy_label='Taxonomy', others_label=None,
    color_zero_bin=False
):
    """
    Bar plot for each integer-valued feature, colored by taxonomy (or group label), with fixed bar width.
    Allows custom taxonomy_label for legend and grouping.
    Optionally skips coloring the zero bin by taxonomy.
    """
    if others_label is None:
        others_label = f'Other {taxonomy_label.lower()}'

    def get_n_colors(n):
        if n <= 20:
            return plt.get_cmap(palette).colors[:n]
        else:
            cmap = cm.get_cmap('nipy_spectral', n)
            return [cmap(i) for i in range(n)]

    sample_taxa = feature_df.index.map(tax_dict.get)
    taxa_counts = Counter(sample_taxa.dropna())
    if count_threshold is not None:
        taxa_above = [tax for tax, count in taxa_counts.items() if count >= count_threshold]
        sample_taxa = sample_taxa.where(sample_taxa.isin(taxa_above), others_label)
        taxa_counts = Counter(sample_taxa.dropna())
    if max_taxa is not None:
        top_taxa = [tax for tax, count in taxa_counts.most_common(max_taxa)]
        if others_label in taxa_counts and others_label not in top_taxa:
            top_taxa.append(others_label)
    else:
        top_taxa = list(taxa_counts.keys())

    n_feats = len(features)
    ncols = 2 if show_cdf else 1
    fig, axs = plt.subplots(n_feats, ncols, figsize=(figsize[0]*ncols, figsize[1]*n_feats), squeeze=False)

    fontsize = 15
    ticksize = 12

    for i, feat in enumerate(features):
        data = feature_df[feat]
        data_by_tax = []
        unique_taxa = []
        for tx in top_taxa:
            values = data[sample_taxa == tx].values
            if len(values) > 0:
                data_by_tax.append(values)
                unique_taxa.append(tx)
        n_taxa = len(unique_taxa)
        colors = get_n_colors(n_taxa)
        if len(data_by_tax) == 0:
            print(f"Skipping {feat}: no data for selected taxa.")
            continue

        # Calculate bins and heights as before
        all_values = np.concatenate(data_by_tax)
        minv, maxv = int(np.min(all_values)), int(np.max(all_values))
        value_bins = np.arange(minv, maxv+1)
        heights = np.zeros((n_taxa, len(value_bins)))
        for j, vals in enumerate(data_by_tax):
            counts = Counter(vals)
            for k, v in enumerate(value_bins):
                heights[j, k] = counts.get(v, 0)

        ax = axs[i, 0]
        bottom = np.zeros(len(value_bins))

        # --- Option: gray non-taxonomy for zero bin ---
        if not color_zero_bin and 0 in value_bins:
            k0 = np.where(value_bins == 0)[0][0]
            zero_count = heights[:, k0].sum()
            ax.bar(
                value_bins[k0], zero_count,
                width=bar_width, bottom=0, color='#B0B0B0',
                label='', align='center', edgecolor='black', linewidth=1.1
            )
            bar_start = 1  # start coloring from bin after zero
        else:
            bar_start = 0  # color all bins

        # Now plot colored/stacked bars (skip zero if already drawn)
        for k in range(bar_start, len(value_bins)):
            bottom_bar = 0
            for j in range(n_taxa):
                if heights[j, k] > 0:
                    ax.bar(
                        value_bins[k], heights[j, k],
                        width=bar_width, bottom=bottom_bar,
                        color=colors[j], label=unique_taxa[j] if k == bar_start else "",  # only add to legend once
                        align='center', edgecolor='black', linewidth=1.1
                    )
                    bottom_bar += heights[j, k]

        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicated taxonomy labels, keep order
        seen = set()
        legend_entries = []
        for h, l in zip(handles, labels):
            if l not in seen and l != "":
                seen.add(l)
                legend_entries.append((h, l))
        ax.legend(
            [h for h, l in legend_entries],
            [l for h, l in legend_entries],
            title=taxonomy_label,
            fontsize=11, title_fontsize=12, loc='best'
        )

        ax.set_xlabel(f"Number of {feat} genes", fontsize=fontsize)
        ax.set_ylabel("Number of species", fontsize=fontsize)
        # ax.set_title(f"{feat} Histogram by {taxonomy_label}", fontsize=fontsize+2)
        ax.set_title(f"", fontsize=fontsize+2)
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Optional CDF panel
        if show_cdf:
            ax2 = axs[i, 1]
            for j, tx in enumerate(unique_taxa):
                sorted_data = np.sort(data_by_tax[j])
                cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                ax2.plot(sorted_data, cdf, label=tx, color=colors[j], linewidth=2)
            ax2.set_xlabel(f"{feat} value", fontsize=fontsize)
            ax2.set_ylabel("CDF", fontsize=fontsize)
            #ax2.set_title(f"{feat} CDF by {taxonomy_label}", fontsize=fontsize+2)
            ax2.set_title("", fontsize=fontsize+2)
            ax2.tick_params(axis='both', labelsize=ticksize)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(False)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.legend(title=taxonomy_label, fontsize=11, title_fontsize=12, loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {save_path}")
    plt.show()

def summarize_label_distribution_pub(label_df, bins=50, figsize=(12,5), log_hist=True, show_cdf=True, save_path=None, dpi=300):
    """
    Publication-quality panel: histogram and CDF of label (ingredient) coverage.
    Optionally saves as PNG/PDF.
    """
    # Use consistent, publication-grade styles
    plt.style.use('seaborn-v0_8-whitegrid')
    label_counts = (label_df != 0).sum(axis=0).values
    n_labels = len(label_counts)
    n_samples = label_df.shape[0]

    fig, axs = plt.subplots(1, 2 if show_cdf else 1, figsize=figsize)

    # Set font sizes
    fontsize = 20
    ticksize = 14
    titlesize = 18

    # Histogram
    ax = axs[0] if show_cdf else axs
    ax.hist(label_counts, bins=bins, log=log_hist, edgecolor='black', linewidth=1.2, color="#4F81BD")
    ax.set_xlabel("Number of media containing label", fontsize=fontsize)
    ax.set_ylabel("Number of labels", fontsize=fontsize)
    ax.set_title("Label (Ingredient) Coverage Histogram", fontsize=titlesize, pad=15)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

    # CDF
    if show_cdf:
        label_counts_sorted = np.sort(label_counts)
        yvals = np.arange(1, n_labels+1) / n_labels
        axs[1].plot(label_counts_sorted, yvals, color="#C0504D", linewidth=2)
        axs[1].set_xlabel("Number of media containing label", fontsize=fontsize)
        axs[1].set_ylabel("Cumulative fraction of labels", fontsize=fontsize)
        axs[1].set_title("CDF: Label Coverage", fontsize=titlesize, pad=15)
        axs[1].tick_params(axis='both', labelsize=ticksize)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary stats (can also save to file)
    print("=== Label (Ingredient) Coverage Summary ===")
    print(f"Total labels: {n_labels}")
    print(f"Total samples: {n_samples}")
    print(f"Mean label presence: {label_counts.mean():.2f}")
    print(f"Median label presence: {np.median(label_counts)}")
    print(f"Min/Max label presence: {label_counts.min()} / {label_counts.max()}")
    print(f"Labels present in only one sample: {(label_counts == 1).sum()}")
    print(f"Labels present in ALL samples: {(label_counts == n_samples).sum()}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"{int(q*100)}% of labels present in ≤ {np.percentile(label_counts, q*100):.0f} samples")

def filter_columns_by_count(df, min_count=None, max_count=None, verbose=True, coltype="column"):
    """
    Remove columns based on count of non-zeros per column.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, rows = samples, columns = features or labels.
    min_count : int or None
        Remove columns with less than min_count non-zeros.
    max_count : int or None
        Remove columns with more than max_count non-zeros.
    verbose : bool, default=True
        Print summary of filtering.
    coltype : str, default="column"
        String for print messages: "feature", "label", etc.
    Returns
    -------
    filtered_df : pd.DataFrame
    """
    col_counts = (df != 0).sum(axis=0)
    to_keep = col_counts.index
    if min_count is not None:
        to_keep = to_keep[col_counts >= min_count]
    if max_count is not None:
        to_keep = to_keep[col_counts <= max_count]
    filtered_df = df[to_keep]
    if verbose:
        removed = set(df.columns) - set(to_keep)
        print(f"Filtered out {len(removed)} {coltype}s.")
        if min_count is not None:
            print(f"  - {sum(col_counts < min_count)} {coltype}s had < {min_count} non-zero samples.")
        if max_count is not None:
            print(f"  - {sum(col_counts > max_count)} {coltype}s had > {max_count} non-zero samples.")
        print(f"Remaining {coltype}s: {filtered_df.shape[1]}")
    return filtered_df

def tune_rf_multilabel(
    pfam_df, media_df, test_size=0.2, random_state=42, n_iter=10, verbose=True
):
    pfam_df, media_df = check_and_align_indices(pfam_df, media_df, drop_diff=True, verbose=False)
    X, y = pfam_df, media_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    param_dist = {
        'estimator__n_estimators': [100, 200, 500],
        'estimator__max_depth': [5, 20],
        'estimator__min_samples_split': [2, 5],
        # 'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__class_weight': ['balanced_subsample']
    }

    base_rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    clf = MultiOutputClassifier(base_rf)

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1_micro',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=random_state
    )

    print(f">> Performing RandomizedSearchCV with {n_iter} iterations...")
    search.fit(X_train, y_train)

    print("\nBest parameters:", search.best_params_)
    print("Best f1_micro score (CV): {:.4f}".format(search.best_score_))

    # Collect all search iteration results as a list of dicts
    results = []
    for i in range(len(search.cv_results_['params'])):
        p = search.cv_results_['params'][i]
        d = {
            "Setting": f"Search iter {i+1}",
            "n_samples": X_train.shape[0],
            "n_labels": y_train.shape[1],
            "n_features": X_train.shape[1],
            "n_estimators": p['estimator__n_estimators'],
            "max_depth": p['estimator__max_depth'],
            "min_split": p['estimator__min_samples_split'],
            # "min_samples_leaf": p['estimator__min_samples_leaf'],
            # "class_weight": p['estimator__class_weight'],
            "Mean F1 Micro": f"{search.cv_results_['mean_test_score'][i]:.3f}",
            #"CV Std F1 Micro": f"{search.cv_results_['std_test_score'][i]:.3f}",
            # "Params": str(p)
        }
        results.append(d)

    # Evaluate on test set using the best estimator
    y_pred = search.predict(X_test)
    metrics_dict = collect_metrics_dict(
        setting="Tuned RF, test set",
        y_true=y_test,
        y_pred=y_pred,
        xtest=X_test,
        n_labels=y_test.shape[1],
        n_features=X_test.shape[1],
        n_samples=X_test.shape[0],
        n_estimators=search.best_params_['estimator__n_estimators'],
        best_params=search.best_params_
    )
    # results.append(metrics_dict)  # Append test set result as final row

    if verbose:
        print("\n=== Test Set Metrics ===")
        print("Accuracy:", metrics_dict["Accuracy"])
        print("F1 Micro:", metrics_dict["F1 Micro"])
        print("F1 Macro:", metrics_dict["F1 Macro"])

    return search.best_estimator_, X_test, y_test, y_pred, results

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def plot_per_label_metric(y_true, y_pred, label_names=None, metric="f1", 
                         top_n=30, figsize=(15, 5), sort_desc=True, 
                         save_path=None, dpi=300, color="#4F81BD"):
    """
    Plot per-label F1, precision, or recall as a bar plot (publication quality).
    
    Parameters
    ----------
    y_true : array-like or DataFrame
        Ground truth label matrix (n_samples, n_labels)
    y_pred : array-like or DataFrame
        Predicted label matrix (n_samples, n_labels)
    label_names : list-like, optional
        Names of the labels (columns); required for nice axis labels.
    metric : str, default="f1"
        Which metric to plot: "f1", "precision", or "recall"
    top_n : int, default=30
        Plot only the top_n labels by metric value (to avoid overcrowding).
    figsize : tuple, default=(15, 5)
        Figure size.
    sort_desc : bool, default=True
        Whether to sort bars by metric value descending.
    save_path : str, optional
        If set, save figure to this path.
    dpi : int, default=300
        Dots per inch for saved figure.
    color : str, default="#4F81BD"
        Bar color.
    """
    if label_names is None:
        label_names = [str(i) for i in range(y_true.shape[1])]
        
    # Select metric function
    metric_func = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score
    }[metric]
    
    scores = metric_func(y_true, y_pred, average=None, zero_division=0)
    scores_series = pd.Series(scores, index=label_names)
    
    if sort_desc:
        scores_series = scores_series.sort_values(ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    scores_series.head(top_n).plot(kind='bar', color=color, edgecolor='black', linewidth=1.2)
    plt.ylabel(f"{metric.title()} Score", fontsize=16)
    plt.xlabel("Label", fontsize=16)
    plt.title(f"Per-label {metric.title()} Scores (Top {top_n})", fontsize=18, pad=15)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    # Annotate bars with score values (optional)
    for i, v in enumerate(scores_series.head(top_n)):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10, rotation=90)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Plot saved to: {save_path}")
    plt.show()
    
    # Print summary stats
    print(f"Mean {metric} (across labels): {scores_series.mean():.3f}")
    print(f"Median {metric}: {scores_series.median():.3f}")
    print(f"Min/Max {metric}: {scores_series.min():.3f} / {scores_series.max():.3f}")
    return scores_series


def plot_feature_importance(
    clf, feature_names, top_n=30, figsize=(15, 5),
    color="#C0504D", save_path=None, dpi=300
):
    """
    Plot top-N feature importances from a MultiOutputClassifier (e.g., Random Forest).
    
    Parameters
    ----------
    clf : MultiOutputClassifier
        Fitted classifier (must have .estimators_ attribute).
    feature_names : list or Index
        Feature (Pfam) names, length must match input features.
    top_n : int, default=30
        Show only top-N features by mean importance.
    figsize : tuple, default=(15, 5)
        Figure size.
    color : str, default="#C0504D"
        Bar color.
    save_path : str, optional
        If set, save to file.
    dpi : int, default=300
        DPI for figure file.
    """
    # Each .estimator_ is a RandomForestClassifier for one label; .feature_importances_ is 1D array per label
    # We'll average across all outputs (labels)
    all_importances = np.array([est.feature_importances_ for est in clf.estimators_])
    mean_importance = all_importances.mean(axis=0)
    importance_series = pd.Series(mean_importance, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    importance_series.head(top_n).plot(
        kind='bar', color=color, edgecolor='black', linewidth=1.2
    )
    plt.ylabel("Mean Feature Importance", fontsize=16)
    plt.xlabel("Feature (Pfam domain)", fontsize=16)
    plt.title(f"Top {top_n} Feature Importances (Mean across labels)", fontsize=18, pad=15)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Annotate bars with value
    for i, v in enumerate(importance_series.head(top_n)):
       plt.text(i, v + 0.02 * importance_series.head(top_n).max(), f"{v:.3f}",
         ha='center', va='bottom', fontsize=10, rotation=90)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Print a mini-table of importances
    print("=== Top Feature Importances ===")
    print(importance_series.head(top_n).to_string())
    return importance_series

def plot_feature_importance_pct(
    clf, feature_names, top_n=30, figsize=(20, 8),
    color="#C0504D", save_path=None, dpi=300
):
    """
    Plot top-N feature importances (as %) from a MultiOutputClassifier (e.g., Random Forest).
    """
    all_importances = np.array([est.feature_importances_ for est in clf.estimators_])
    mean_importance = all_importances.mean(axis=0)
    importance_pct = mean_importance * 100  # convert to percent
    importance_series = pd.Series(importance_pct, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    importance_series.head(top_n).plot(
        kind='bar', color=color, edgecolor='black', linewidth=1.2
    )
    plt.ylabel("Mean Feature Importance (%)", fontsize=16)
    plt.xlabel("Feature (Pfam domain)", fontsize=16)
    plt.title(f"Top {top_n} Feature Importances (%)", fontsize=18, pad=15)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Annotate bars with value as percentage
    for i, v in enumerate(importance_series.head(top_n)):
        plt.text(i, v + 0.02 * importance_series.head(top_n).max(), f"{v:.3f}",
         ha='center', va='bottom', fontsize=10, rotation=90)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Plot saved to: {save_path}")

    plt.show()

    print("=== Top Feature Importances (%) ===")
    print(importance_series.head(top_n).round(2).to_string())
    return importance_series

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_by_group(
    feature_df, sample_to_group, title="t-SNE Embedding by Group", group_label="Group",
    perplexity=None, figsize=(15,9), save_path=None, dpi=600, random_state=42, s=70, alpha=0.85
):
    feature_df.index = feature_df.index.map(str)
    group_series = pd.Series(feature_df.index.map(sample_to_group), index=feature_df.index)
    valid_mask = group_series.notna()
    if not valid_mask.all():
        print(f"Warning: {(~valid_mask).sum()} samples dropped (no mapping found).")
    X = feature_df.loc[valid_mask].values
    groups = group_series[valid_mask].astype(str).values
    print("Number of unique groups:", len(np.unique(groups)))

    n_samples = X.shape[0]
    if perplexity is None:
        perplexity = min(30, max(2, n_samples // 5))
        if n_samples <= 10:
            perplexity = max(2, n_samples // 2)
    if n_samples <= perplexity:
        raise ValueError(f"Perplexity ({perplexity}) must be less than n_samples ({n_samples}).")

    print(f"Running t-SNE on {n_samples} samples with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)

    plt.style.use('seaborn-v0_8-whitegrid')
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups <= 10:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(n_groups)]
    elif n_groups <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(n_groups)]
    else:
        cmap = plt.get_cmap('hsv')
        colors = [cmap(x) for x in np.linspace(0, 1, n_groups)]

    plt.figure(figsize=figsize)
    for i, grp in enumerate(unique_groups):
        idx = groups == grp
        plt.scatter(
            X_tsne[idx, 0], X_tsne[idx, 1],
            label=f"{grp} (n={idx.sum()})",
            color=colors[i],
            s=22,              # Smaller dots
            alpha=alpha,
            edgecolor='none'   # No edge color
        )
    plt.title(title, fontsize=20, pad=14)
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(
        title=group_label,
        fontsize=12,
        title_fontsize=13,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True
    )
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(rect=[0, 0, 0.80, 1])  # Extra right margin for legend
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()
    return X_tsne, groups


from sklearn.metrics import roc_curve, auc

def plot_multilabel_roc_from_clf(
    clf, X_test, y_test, label_names=None, max_curves=6,
    figsize=(8,8), save_path=None, dpi=600, sort_by_auc=False, curve_indices=None
):
    """
    Plots ROC curves for multilabel classification directly from classifier and X_test.
    Optionally sorts or selects specific label indices to plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    y_test = np.array(y_test)
    n_labels = y_test.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]
    # Calculate probabilities
    if hasattr(clf, "estimators_"):
        y_score = np.column_stack([
            est.predict_proba(X_test)[:, 1] for est in clf.estimators_
        ])
    else:
        y_score = clf.predict_proba(X_test)
        if y_score.ndim == 3:
            y_score = y_score[:,:,1]
    
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_labels):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot micro-average
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
        color='deeppink', linestyle=':', linewidth=3,
    )

    # --- NEW: Use user-provided indices if given ---
    if curve_indices is not None:
        indices = list(curve_indices)
    else:
        indices = list(range(n_labels))
        if sort_by_auc:
            indices = sorted(indices, key=lambda i: roc_auc[i], reverse=True)
        indices = indices[:max_curves]

    # Plot requested ROC curves
    for idx in indices:
        plt.plot(
            fpr[idx], tpr[idx],
            lw=2, label=f"{label_names[idx]} (AUC = {roc_auc[idx]:.2f})"
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves (multi-label)', fontsize=17)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_label_correlation_heatmap(
    media_df,
    top_n=20,
    labels=None,
    method='pearson',
    figsize=(10, 8),
    cmap='coolwarm',
    annot=False,
    save_path=None,
    dpi=600,
    label_truncate=14,         # NEW: Max chars to show
    xtick_rotation=45,         # NEW: Angle of x labels
    truncate_y=True            # NEW: Also truncate y-axis labels?
):
    """
    Plot a heatmap of pairwise correlations among selected or top-N most common labels.
    """
    # Determine which labels to use
    if labels is not None:
        label_cols = [lbl for lbl in labels if lbl in media_df.columns]
        if len(label_cols) == 0:
            raise ValueError("None of the specified labels found in DataFrame columns.")
        if len(label_cols) < len(labels):
            missing = set(labels) - set(label_cols)
            print(f"Warning: The following labels were not found and will be skipped: {missing}")
    else:
        label_cols = media_df.sum().sort_values(ascending=False).head(top_n).index.tolist()
    
    sub_df = media_df[label_cols]
    corr = sub_df.corr(method=method)
    
    # Truncate labels for display
    def trunc(lab):
        return (lab[:label_truncate] + "…") if len(lab) > label_truncate else lab
    xtick_labels = [trunc(lbl) for lbl in label_cols]
    ytick_labels = [trunc(lbl) for lbl in label_cols] if truncate_y else label_cols
    
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    im = plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.grid(False)
    plt.xticks(
        ticks=np.arange(len(label_cols)), 
        labels=xtick_labels, 
        rotation=xtick_rotation, 
        fontsize=13, ha='right'
    )
    plt.yticks(
        ticks=np.arange(len(label_cols)), 
        labels=ytick_labels, 
        fontsize=13
    )
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f'{method.capitalize()} correlation')
    title_labels = f"labels: {', '.join(label_cols)}" if labels is not None else f"top {top_n} labels"
    plt.title(f'Label Correlation Heatmap ({title_labels})', fontsize=17, pad=15)

    # Annotate values
    if annot and len(label_cols) <= 15:
        for i in range(len(label_cols)):
            for j in range(len(label_cols)):
                val = corr.iloc[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()
    return corr


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_multilabel_confusion_matrix(
    y_true, y_pred, labels,
    class_names=('Negative', 'Positive'),
    figsize=(5, 5),
    cmap='Blues',
    save_path=None,
    dpi=600
):
    """
    Plot confusion matrix for specified labels in multilabel data.

    Parameters
    ----------
    y_true : DataFrame or array-like
        True binary labels (columns = all possible labels).
    y_pred : DataFrame or array-like
        Predicted binary labels (same structure as y_true).
    labels : list of str
        List of label names to plot confusion matrices for.
    class_names : tuple
        Class labels for axes (default: ('Negative', 'Positive')).
    figsize : tuple
        Figure size per matrix.
    cmap : str
        Colormap for confusion matrix.
    save_path : str, optional
        Path to save the figure.
    dpi : int
        DPI for saved figure.
    """
    # If DataFrame, select columns
    if hasattr(y_true, "columns"):
        y_true = y_true[labels].values
    else:
        # Assume correct column order in array
        y_true = np.array(y_true)
    if hasattr(y_pred, "columns"):
        y_pred = y_pred[labels].values
    else:
        y_pred = np.array(y_pred)
    
    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(figsize[0]*n_labels, figsize[1]))
    if n_labels == 1:
        axes = [axes]

    plt.style.use('seaborn-v0_8-whitegrid')
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0)
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f"{val}", ha='center', va='center', color='black', fontsize=14)
        ax.set_title(label, fontsize=16)
        ax.set_xlabel('Predicted', fontsize=13)
        ax.set_ylabel('True', fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()

def plot_pfam_prevalence_distribution(
    pfam_df,
    bins=50,
    figsize=(10,6),
    save_path=None,
    dpi=600
):
    """
    Plot the prevalence distribution for Pfam domains (fraction of samples present).

    Parameters
    ----------
    pfam_df : pd.DataFrame
        Pfam matrix (rows = samples, columns = Pfam domains; binary or counts).
    bins : int
        Number of bins for the histogram.
    figsize : tuple
        Size of the figure.
    save_path : str, optional
        If set, save the figure here.
    dpi : int
        DPI for saving the figure.
    """
    # Compute prevalence for each Pfam domain
    prevalence = (pfam_df > 0).sum(axis=0) / pfam_df.shape[0]

    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.hist(prevalence, bins=bins, edgecolor='black')
    plt.grid(False)
    plt.xlabel('Prevalence (Fraction of Samples Present)', fontsize=15)
    plt.ylabel('Number of Pfam Domains', fontsize=15)
    plt.title('Pfam Domain Prevalence Distribution', fontsize=17, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()

    # Print summary
    print(f"Total Pfam domains: {len(prevalence)}")
    print(f"Median Pfam prevalence: {np.median(prevalence):.3f}")
    print(f"Pfam domains present in ≤5% of samples: {(prevalence <= 0.05).sum()}")
    print(f"Pfam domains present in ≥50% of samples: {(prevalence >= 0.5).sum()}")
    return prevalence

def plot_pfam_prevalence_curve(
    pfam_df,
    figsize=(14, 6),
    color='#2C72B7',
    save_path=None,
    dpi=600
):
    """
    Plot a curve of Pfam domain prevalence (fraction of genomes containing each domain), sorted.

    Parameters
    ----------
    pfam_df : pd.DataFrame
        Pfam matrix (rows = samples, columns = Pfam domains; binary or counts).
    figsize : tuple
        Figure size.
    color : str
        Color for the curve.
    save_path : str, optional
        If set, save the figure here.
    dpi : int
        DPI for saving the figure.
    """
    # Calculate prevalence for each domain
    prevalence = (pfam_df > 0).sum(axis=0) / pfam_df.shape[0]
    prevalence_sorted = np.sort(prevalence)[::-1]  # Descending order

    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(np.arange(1, len(prevalence_sorted) + 1), prevalence_sorted, color=color, linewidth=2.2)
    plt.grid(False)
    plt.xlabel('Pfam Domains (sorted by prevalence)', fontsize=15)
    plt.ylabel('Fraction of Genomes Containing Domain', fontsize=15)
    plt.title('Pfam Prevalence Curve', fontsize=18, pad=13)
    plt.ylim(-0.01, 1.01)
    plt.xlim(1, len(prevalence_sorted))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()

    # Quick stats printout
    print(f"Total Pfam domains: {len(prevalence)}")
    print(f"Median prevalence: {np.median(prevalence):.3f}")
    print(f"Domains present in ≥50% of genomes: {(prevalence >= 0.5).sum()}")
    print(f"Domains present in ≤5% of genomes: {(prevalence <= 0.05).sum()}")
    return prevalence

def plot_feature_importance_for_label(
    clf,                        # MultiOutputClassifier (fitted)
    feature_names,              # list or Index: features
    label_names,                # list of all label names (order must match .estimators_)
    target_label,               # label to plot (e.g. 'Glucose')
    top_n=10,
    color='#C0504D',
    figsize=(12, 8),
    save_path=None,
    dpi=600
):
    """
    Plot top-N feature importances for a specific output label (e.g., 'Glucose'),
    as percentages (summing to 100%), without value annotations.
    """
    if hasattr(clf, "estimators_"):
        if target_label not in label_names:
            raise ValueError(f"Label '{target_label}' not found in label_names!")
        idx = list(label_names).index(target_label)
        estimator = clf.estimators_[idx]
        importances = estimator.feature_importances_
    else:
        estimator = clf
        importances = clf.feature_importances_
    # Convert importances to percentages
    importances_pct = 100 * (importances / importances.sum())
    importance_series = pd.Series(importances_pct, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.bar(
        importance_series.head(top_n).index,
        importance_series.head(top_n).values,
        color=color, edgecolor='black', linewidth=1.15
    )
    plt.grid(False)
    plt.ylabel("Feature Importance (%)", fontsize=15)
    plt.xlabel("Feature (Pfam domain)", fontsize=15)
    plt.title(f"Top {top_n} Feature Importances for '{target_label}'", fontsize=17, pad=13)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=13)
    max_val = importance_series.head(top_n).max()
    plt.ylim(0, max_val * 1.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Plot saved to: {save_path}")
    plt.show()

    print(f"=== Top Features for '{target_label}' (as %) ===")
    print(importance_series.head(top_n).to_string(float_format="%.2f"))
    return importance_series.head(top_n)

from sklearn.metrics import precision_score, recall_score, f1_score

def table_figure_label_metrics(
    y_true, y_pred, label_names, top_n=10, labels=None,
    least=False,
    sort_by_f1=False,       # NEW
    figsize=(8, 0.7), fontsize=16, header_fontsize=17,
    label_col_width=0.45, other_col_width=0.18,
    save_path=None, dpi=600
):
    """
    Generate a publication-quality table figure of precision, recall, F1 for a specified label list,
    or for the top-N labels with highest (or lowest, if least=True) F1-score or prevalence.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score

    # If DataFrames, convert to arrays and get columns
    if hasattr(y_true, 'columns'):
        cols = list(y_true.columns)
        y_true = y_true.values
    else:
        cols = list(label_names)
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    if labels is not None:
        idx = [cols.index(lbl) for lbl in labels if lbl in cols]
        if len(idx) == 0:
            raise ValueError("None of the specified labels found in columns.")
        label_list = [cols[i] for i in idx]
    else:
        # Compute metrics for all labels first
        prec_all = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_all  = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_all   = f1_score(y_true, y_pred, average=None, zero_division=0)

        if sort_by_f1:
            if least:
                idx = np.argsort(f1_all)[:top_n]
            else:
                idx = np.argsort(-f1_all)[:top_n]
            label_list = [cols[i] for i in idx]
        else:
            prevalence = y_true.sum(axis=0)
            if least:
                idx = np.argsort(prevalence)[:top_n]
                label_list = [cols[i] for i in idx]
            else:
                idx = np.argsort(-prevalence)[:top_n]
                label_list = [cols[i] for i in idx]
        # Now restrict precision, recall, f1 to the chosen indices
        prec = prec_all[idx]
        rec = rec_all[idx]
        f1 = f1_all[idx]
    # If using custom label list, compute metrics only for these
    if labels is not None:
        prec = precision_score(y_true[:, idx], y_pred[:, idx], average=None, zero_division=0)
        rec = recall_score(y_true[:, idx], y_pred[:, idx], average=None, zero_division=0)
        f1 = f1_score(y_true[:, idx], y_pred[:, idx], average=None, zero_division=0)

    df = pd.DataFrame({
        "Label": label_list,
        "Precision": np.round(prec, 3),
        "Recall": np.round(rec, 3),
        "F1-score": np.round(f1, 3)
    })

    # Figure and table
    nrows = df.shape[0] + 1
    ncols = df.shape[1]
    cell_height = figsize[1]
    plt.figure(figsize=(figsize[0], nrows * cell_height))
    plt.axis('off')
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Set header font weight and font size, and column widths
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_fontsize(header_fontsize)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E1EAF2')
        cell.set_height(cell_height)
        cell.set_linewidth(0.5)
        if col == 0:
            cell.set_width(label_col_width)
        else:
            cell.set_width(other_col_width)

    # Title
    if labels is not None:
        table_title = f"Precision, Recall, F1 for Selected Labels"
    elif sort_by_f1:
        which = "Lowest" if least else "Highest"
        table_title = f"Precision, Recall, F1 for {top_n} Labels with {which} F1-score"
    elif least:
        table_title = f"Precision, Recall, F1 for {len(df)} Least Prevalent Ingredients"
    else:
        table_title = f"Precision, Recall, F1 for Top {len(df)} Most Prevalent Ingredients"
    plt.title(table_title, fontsize=fontsize+4, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Table figure saved to: {save_path}")
    plt.show()
    return df

def plot_violin_label_metrics(
    y_true, y_pred, label_names=None,
    figsize=(7,6), fontsize=19,
    colors=["#46627f", "#7e9489", "#b09c6d"],  # Professional muted
    save_path=None, dpi=600,
    show_stats=True
):
    """
    Publication-quality violin plot for per-label Precision, Recall, and F1 with professional muted colors.
    """
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    data = [precisions, recalls, f1s]
    metric_names = ['Precision', 'Recall', 'F1']

    plt.style.use('seaborn-v0_8-whitegrid')  # Smooth background

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        data, showmeans=True, showmedians=True, showextrema=False
    )

    # Set custom muted colors and alpha for each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.83)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.2)
    # Set mean and median style
    parts['cmeans'].set_edgecolor('#444444')
    parts['cmeans'].set_linewidth(2.5)
    parts['cmedians'].set_edgecolor('#222222')
    parts['cmedians'].set_linewidth(2.7)

    ax.set_xticks([1,2,3])
    ax.set_xticklabels(metric_names, fontsize=fontsize)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)], fontsize=fontsize-2)
    ax.set_ylabel("Score", fontsize=fontsize+1, labelpad=9)
    ax.set_xlabel("Metric", fontsize=fontsize+1, labelpad=9)
    ax.set_title("Distribution of Per-Label Metrics", fontsize=fontsize+4, pad=16)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', which='both', width=2, length=5)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    plt.show()

    if show_stats:
        for arr, name in zip(data, metric_names):
            print(f"{name}  Mean: {np.mean(arr):.3f}  |  Median: {np.median(arr):.3f}")

    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1s
    }