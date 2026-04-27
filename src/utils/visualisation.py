import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from pathlib import Path


# Internal helper

def _prepare_output_dir(model_name):
    base_dir = Path("outputs/figures") / model_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _save_fig(fig, path, dpi=300):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)



#core metric aggregation
def aggregate_threshold_metrics(thresholds, f1_scores, precision_scores, recall_scores):
    """Core Metric Aggregation Function.
    Supports dict-based (thr -> list), list/array-based (index-aligned with thresholds),
    or list-of-lists where each index corresponds to a threshold.
    """
    rows = []
    for idx, thr in enumerate(thresholds):
        # flexible input handling: dict (thr -> list), list/array aligned with thresholds,
        # or list-of-lists where each index corresponds to a threshold
        def _get_stats(obj):
            if obj is None:
                return (np.nan, np.nan)
            if isinstance(obj, dict):
                vals = np.asarray(obj.get(thr, []), dtype=float)
            else:
                arr = np.asarray(obj)
                if arr.ndim == 1 and len(arr) == len(thresholds):
                    vals = np.atleast_1d(arr[idx]).astype(float)
                elif arr.ndim >= 2:
                    vals = np.asarray(arr[idx], dtype=float)
                else:
                    vals = arr.astype(float)
            if vals.size == 0:
                return (np.nan, np.nan)
            return (np.nanmean(vals), np.nanstd(vals))
        f1_mean, f1_std = _get_stats(f1_scores)
        p_mean, p_std = _get_stats(precision_scores)
        r_mean, r_std = _get_stats(recall_scores)
        rows.append({
            "Threshold": thr,
            "F1 Mean": f1_mean,
            "F1 Std": f1_std,
            "Precision Mean": p_mean,
            "Precision Std": p_std,
            "Recall Mean": r_mean,
            "Recall Std": r_std,
        })
    return pd.DataFrame(rows).set_index("Threshold")


#heatmap
def plot_threshold_heatmap(df_metrics, save_dir=None, show=True):
    fig = plt.figure(figsize=(8, 5))
    sns.heatmap(
        df_metrics[["F1 Mean", "Precision Mean", "Recall Mean"]],
        annot=True, cmap="YlGnBu", fmt=".3f"
    )
    plt.title("Performance Heatmap by Threshold")
    plt.tight_layout()
    if save_dir is not None:
        path = Path(save_dir) / "threshold_heatmap.png"
        _save_fig(fig, path)
    if show:
        plt.show()


# probability distributions
def plot_probability_distributions(y_prob, y_test_int, threshold=0.4, save_dir=None, show=True):
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(np.asarray(y_prob)[np.asarray(y_test_int) == 0], color="red", label="Abnormal (Hypoxia)", kde=True, bins=10)
    sns.histplot(np.asarray(y_prob)[np.asarray(y_test_int) == 1], color="green", label="Normal", kde=True, bins=10)
    plt.axvline(threshold, color='black', linestyle='--', label=f"Suggested Threshold ({threshold})")
    plt.title("Predicted Probability Distribution")
    plt.xlabel("P(Normal)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        _save_fig(fig, Path(save_dir) / "probability_distributions.png")
    if show:
        plt.show()


# roc curves
def plot_roc_curves(y_true_list, y_prob_list, save_dir=None, show=True):
    plt.figure(figsize=(7, 6))
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []
    for y_true, y_prob in zip(y_true_list, y_prob_list):
        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_prob))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, color='grey', alpha=0.3)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs) if len(aucs) > 0 else np.nan
    std_auc = np.std(aucs) if len(aucs) > 0 else np.nan
    plt.plot(base_fpr, mean_tpr, color='b', lw=2, label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All folds)")
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    if save_dir is not None:
        _save_fig(plt.gcf(), Path(save_dir) / "roc_curves.png")
    if show:
        plt.show()
    return mean_auc, std_auc


#precision-recall 
def plot_precision_recall_curve_all(y_true_list, y_prob_list, save_dir=None, show=True):
    all_y_true = np.concatenate([np.asarray(y) for y in y_true_list])
    all_y_prob = np.concatenate([np.asarray(p) for p in y_prob_list])
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    ap = average_precision_score(all_y_true, all_y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (All CV folds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if save_dir is not None:
        _save_fig(fig, Path(save_dir) / "precision_recall_curve.png")
    if show:
        plt.show()
    return ap


#confusion matrix
def plot_confusion_matrix(y_true, y_prob, threshold=0.5, save_dir=None, show=True):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true), y_pred)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix (thr={threshold})")
    plt.tight_layout()
    if save_dir is not None:
        _save_fig(fig, Path(save_dir) / f"confusion_matrix_thr_{threshold:.2f}.png")
    if show:
        plt.show()
    return cm


# save tables
def save_metric_table(df, path="outputs/tables/threshold_metrics.csv"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return str(p)


#one-line master function
def generate_all_plots_and_tables(
    thresholds,
    f1_scores,
    precision_scores,
    recall_scores,
    y_prob,
    y_test_int,
    y_true_folds,
    y_prob_folds,
    model_name="model",
    suggested_threshold=0.4,
    save_tables=False,
    save_path="outputs/tables/threshold_metrics.csv",
    show=True
):
    save_dir = _prepare_output_dir(model_name)
    df_metrics = aggregate_threshold_metrics(thresholds, f1_scores, precision_scores, recall_scores)
    # persist table
    if save_tables:
        save_metric_table(df_metrics, save_path)
    # plots
    plot_threshold_heatmap(df_metrics, save_dir=save_dir, show=show)
    plot_probability_distributions(y_prob, y_test_int, threshold=suggested_threshold, save_dir=save_dir, show=show)
    ap = plot_precision_recall_curve_all(y_true_folds, y_prob_folds, save_dir=save_dir, show=show)
    mean_auc, std_auc = plot_roc_curves(y_true_folds, y_prob_folds, save_dir=save_dir, show=show)
    cm = plot_confusion_matrix(y_test_int, y_prob, threshold=suggested_threshold, save_dir=save_dir, show=show)
    return {
        "threshold_metrics": df_metrics,
        "ap": ap,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "confusion_matrix": cm,
        "figure_dir": save_dir
    }
