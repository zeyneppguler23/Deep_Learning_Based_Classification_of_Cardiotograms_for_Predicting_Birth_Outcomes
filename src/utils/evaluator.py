import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)

class Evaluator:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def evaluate(self, y_true, y_prob):
        """Evaluate predictions and return metrics for all thresholds."""
        results = {"auc": roc_auc_score(y_true, y_prob), "thresholds": {}}

        for thr in self.thresholds:
            y_pred = (y_prob >= thr).astype(int)
            results["thresholds"][thr] = {
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "sensitivity": recall_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred)
            }
        return results

    def evaluate_with_print(self, y_true, y_prob, verbose=True):
        """
        Evaluates abnormal detection performance.
        y_true: integer labels (1 if abnormal, 0 if normal)
        y_prob: probability of abnormal (class 0)
        """

        if verbose:
            print(f"Predictions: min={y_prob.min():.3f}, max={y_prob.max():.3f}, "
                  f"mean={y_prob.mean():.3f}, std={y_prob.std():.3f}")

        # AUC (exact notebook logic)
        try:
            auc = roc_auc_score(y_true, y_prob)
            if verbose:
                print(f"AUC: {auc:.3f}")
        except Exception as e:
            print(f"✗ AUC error: {e}")
            auc = np.nan

        fold_results = {
            "auc": auc,
            "thresholds": {},
            "y_true": y_true,
            "y_prob": y_prob
        }

        for thr in self.thresholds:
            y_pred = (y_prob >= thr).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            sens = rec
            acc = accuracy_score(y_true, y_pred)

            fold_results["thresholds"][thr] = {
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "sensitivity": sens,
                "accuracy": acc
            }

            if verbose:
                print(
                    f"Thr: {thr:.3f} | "
                    f"F1: {f1:.3f} | "
                    f"Prec: {prec:.3f} | "
                    f"Rec: {rec:.3f} | "
                    f"Sens: {sens:.3f} | "
                    f"Acc: {acc:.3f}"
                )

        return fold_results

    def report_paper_results(self, all_results, threshold=0.4):
        """
        Aggregate results from all folds and report in CTG-Net paper format.
        """
        auc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        sensitivity_scores = []
        for fold_result in all_results:
            auc_scores.append(fold_result.get("auc", np.nan))
            f1_scores.append(fold_result["thresholds"][threshold]["f1"])
            precision_scores.append(fold_result["thresholds"][threshold]["precision"])
            recall_scores.append(fold_result["thresholds"][threshold]["recall"])

        auc_scores = np.array(auc_scores, dtype=float)
        f1_scores = np.array(f1_scores, dtype=float)
        precision_scores = np.array(precision_scores, dtype=float)
        recall_scores = np.array(recall_scores, dtype=float)

        # filter NaNs (e.g. folds where AUC couldn't be computed)
        valid_auc_mask = ~np.isnan(auc_scores)
        if not np.any(valid_auc_mask):
            print("Warning: all AUC values are NaN; check folds for single-class y_true or invalid probabilities.")
            auc_mean = auc_std = np.nan
        else:
            auc_mean = np.mean(auc_scores[valid_auc_mask])
            auc_std = np.std(auc_scores[valid_auc_mask])

        results = {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
        }

        print("\n===== CTG-Net 10-Fold CV Results =====")
        print(f"Mean AUC:       {results['auc_mean']:.3f} ± {results['auc_std']:.3f}" if not np.isnan(results['auc_mean']) else "Mean AUC:       NaN")
        print(f"Mean F1:        {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
        print(f"Mean Precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")
        print(f"Mean Recall:    {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")

        return results