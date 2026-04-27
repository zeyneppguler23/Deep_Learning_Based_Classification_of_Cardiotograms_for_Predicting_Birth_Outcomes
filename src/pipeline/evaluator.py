import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)

class BinaryClassifierEvaluator:
    def __init__(self, thresholds=(0.2, 0.3, 0.4, 0.5)):
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self.fold_results = []
        self.all_y_true = []
        self.all_y_prob = []

    def evaluate_fold(self, y_true, y_prob, iteration, fold):
        fold_result = {
            "iteration": iteration,
            "fold": fold,
            "auc": roc_auc_score(y_true, y_prob),
        }

        for thr in self.thresholds:
            y_pred = (y_prob >= thr).astype(int)
            fold_result[f"f1@{thr}"] = f1_score(y_true, y_pred, zero_division=0)
            fold_result[f"precision@{thr}"] = precision_score(y_true, y_pred, zero_division=0)
            fold_result[f"recall@{thr}"] = recall_score(y_true, y_pred, zero_division=0)
            fold_result[f"sensitivity@{thr}"] = recall_score(y_true, y_pred, zero_division=0)
            fold_result[f"accuracy@{thr}"] = accuracy_score(y_true, y_pred)

        self.fold_results.append(fold_result)
        self.all_y_true.extend(y_true)
        self.all_y_prob.extend(y_prob)

        return fold_result

    # Aggregation 
    def aggregate(self):
        df = pd.DataFrame(self.fold_results)
        summary = {}

        summary["auc_mean"] = df["auc"].mean()
        summary["auc_std"] = df["auc"].std()

        for thr in self.thresholds:
            for metric in ["f1", "precision", "recall", "sensitivity", "accuracy"]:
                col = f"{metric}@{thr}"
                summary[f"{metric}@{thr}_mean"] = df[col].mean()
                summary[f"{metric}@{thr}_std"] = df[col].std()

        return summary, df

    #  Paper-style Reporting 
    def report_paper_results(self, threshold=0.4):
        summary, _ = self.aggregate()
        print("\n===== CTG-Net 10-Fold CV Results =====")
        print(f"Mean AUC:        {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
        print(f"Mean F1:         {summary[f'f1@{threshold}_mean']:.3f} ± {summary[f'f1@{threshold}_std']:.3f}")
        print(f"Mean Precision:  {summary[f'precision@{threshold}_mean']:.3f}")
        print(f"Mean Recall:     {summary[f'recall@{threshold}_mean']:.3f}")
        print(f"Mean Sensitivity:{summary[f'sensitivity@{threshold}_mean']:.3f}")
        print(f"Mean Accuracy:   {summary[f'accuracy@{threshold}_mean']:.3f}")

        return summary

    #  Logging 
    def save_csv(self, path="results/fold_results.csv"):
        df = pd.DataFrame(self.fold_results)
        df.to_csv(path, index=False)
        print(f"✓ Saved fold-level results → {path}")

    def save_json(self, path="results/summary.json"):
        summary, _ = self.aggregate()
        import json
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Saved summary results → {path}")
