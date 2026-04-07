from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
import nbformat
import numpy as np
import tensorflow as tf


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


NOTEBOOK_PATH = Path(__file__).with_name("ten_min_segment_pipeline_v10.4.ipynb")
NOTEBOOK_DIR = NOTEBOOK_PATH.parent
OUTPUT_ROOT = NOTEBOOK_DIR / "outputs" / "class_weight_sweep_repeated5x3"
BASELINE_SUMMARY_PATH = NOTEBOOK_DIR / "outputs" / "step23_softmax_repeated5x3" / "step23_softmax_results.json"
N_DEFINITION_CODE_CELLS = 10

WEIGHT_GRID = [
    (0.9, 1.1, 2.3),
    (0.9, 1.2, 2.3),
    (0.9, 1.3, 2.3),
    (0.9, 1.2, 2.5),
    (0.9, 1.3, 2.5),
    (1.0, 1.2, 2.3),
    (1.0, 1.2, 2.5),
]


def load_notebook_namespace() -> dict:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code_cells) < N_DEFINITION_CODE_CELLS:
        raise RuntimeError(
            f"Notebook only contains {len(code_cells)} code cells, expected at least {N_DEFINITION_CODE_CELLS}."
        )

    namespace: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(NOTEBOOK_DIR)
    try:
        for index, cell in enumerate(code_cells[:N_DEFINITION_CODE_CELLS], start=1):
            print(f"\n--- Executing definition cell {index}/{N_DEFINITION_CODE_CELLS} ({cell.get('id')}) ---")
            exec(cell["source"], namespace)
    finally:
        os.chdir(cwd)

    return namespace


def build_repeated_grouped_folds(ns: dict, metadata, cfg: dict):
    n_splits = int(cfg["n_splits"])
    n_repeats = int(cfg.get("n_repeats", 1))
    requested_min_severe = int(cfg.get("min_severe_records", 10))
    repeat_seed_stride = int(cfg.get("repeat_seed_stride", 1000))

    fold_plan = []
    effective_thresholds = []

    for repeat_idx in range(n_repeats):
        repeat_seed = int(cfg["cv_seed"]) + repeat_idx * repeat_seed_stride
        print(f"\nPlanning repeat {repeat_idx + 1} / {n_repeats} (seed={repeat_seed})")
        repeat_folds, effective_min = ns["stratified_record_kfold_with_constraint"](
            metadata,
            n_splits=n_splits,
            min_severe_records=requested_min_severe,
            max_attempts=cfg.get("max_cv_attempts", 100),
            cv_seed=repeat_seed,
        )
        if not repeat_folds:
            raise RuntimeError(
                f"Failed to generate grouped folds for repeat {repeat_idx + 1} with seed {repeat_seed}."
            )

        effective_thresholds.append(int(effective_min))
        for fold_idx, (train_idx, val_idx) in enumerate(repeat_folds, start=1):
            fold_plan.append(
                {
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx,
                    "global_fold": len(fold_plan) + 1,
                    "cv_seed": repeat_seed,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                }
            )

    effective_min_severe = min(effective_thresholds) if effective_thresholds else requested_min_severe
    return fold_plan, effective_min_severe


def install_plot_overrides(ns: dict) -> None:
    def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
        class_names = class_names or ns["CLASS_NAMES"]
        _, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_fold_summary(fold_results, save_path=None):
        f1s = [result["macro_f1"] for result in fold_results]
        folds = list(range(1, len(f1s) + 1))
        _, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(folds, f1s, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axhline(
            np.mean(f1s),
            color="red",
            linestyle="--",
            label=f"Mean = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
        )
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_title("Per-Fold Record-Step Macro F1", fontsize=14)
        ax.set_xticks(folds)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    ns["plot_confusion_matrix"] = plot_confusion_matrix
    ns["plot_fold_summary"] = plot_fold_summary


def weights_to_name(weights: tuple[float, float, float]) -> str:
    return "cw_" + "_".join(str(weight).replace(".", "p") for weight in weights)


def summarize_results(results: list[dict], weights: tuple[float, float, float], cfg: dict) -> dict:
    mean_normal = float(np.mean([result["recall_Normal"] for result in results]))
    mean_mild = float(np.mean([result["recall_Mild"] for result in results]))
    mean_severe = float(np.mean([result["recall_Severe"] for result in results]))
    summary = {
        "experiment": "class_weight_sweep_repeated_grouped_cv",
        "manual_class_weights": {0: weights[0], 1: weights[1], 2: weights[2]},
        "n_folds": int(cfg["n_splits"]),
        "n_repeats": int(cfg.get("n_repeats", 1)),
        "total_fold_runs": int(cfg["n_splits"]) * int(cfg.get("n_repeats", 1)),
        "mean_balanced_acc": float(np.mean([result["balanced_accuracy"] for result in results])),
        "std_balanced_acc": float(np.std([result["balanced_accuracy"] for result in results])),
        "mean_macro_f1": float(np.mean([result["macro_f1"] for result in results])),
        "std_macro_f1": float(np.std([result["macro_f1"] for result in results])),
        "mean_accuracy": float(np.mean([result["accuracy"] for result in results])),
        "std_accuracy": float(np.std([result["accuracy"] for result in results])),
        "mean_recall_Normal": mean_normal,
        "mean_recall_Mild": mean_mild,
        "mean_recall_Severe": mean_severe,
        "passes_guardrails": bool(mean_normal >= 0.60 and mean_mild >= 0.58),
        "per_fold": [{key: value for key, value in result.items() if key != "confusion_matrix"} for result in results],
    }
    return summary


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, default=str)


def run_single_experiment(weights: tuple[float, float, float], force: bool = False) -> Path:
    ns = load_notebook_namespace()
    install_plot_overrides(ns)

    experiment_name = weights_to_name(weights)
    experiment_dir = OUTPUT_ROOT / experiment_name
    summary_path = experiment_dir / "results.json"
    if summary_path.exists() and not force:
        print(f"Skipping {experiment_name} because {summary_path} already exists.")
        return summary_path

    cfg = dict(ns["CFG"])
    cfg.update(
        {
            "n_splits": 5,
            "n_repeats": 3,
            "repeat_seed_stride": 1000,
            "min_severe_records": 10,
            "label_smoothing": 0.05,
            "use_class_weights": True,
            "severe_boost": 0.0,
        }
    )

    X = ns["X_all"]
    y = ns["y_all"]
    groups = ns["groups_all"]
    metadata = ns["_ensure_step_keys"](ns["meta_all"].reset_index(drop=True))
    feature_missing_frac = metadata["missing_frac_prefill"].to_numpy(dtype=np.float32)

    clin_all = None
    if cfg.get("use_clinical_features", True):
        print("Computing engineered features from cleaned FHR windows...")
        clin_all = ns["compute_clinical_features"](
            X,
            fs=ns["SAMPLING_RATE"],
            missing_frac_prefill=feature_missing_frac,
        )
        print(f"  Engineered feature matrix: {clin_all.shape}")

    fold_plan, effective_min_severe = build_repeated_grouped_folds(ns, metadata, cfg)
    requested_min_severe = int(cfg.get("min_severe_records", 10))
    total_fold_runs = len(fold_plan)
    fold_results = []
    all_step_y_true = []
    all_step_y_pred = []

    original_resolver = ns["compute_class_weights_from_y"]

    def _manual_resolver(_y_train):
        resolved = {0: float(weights[0]), 1: float(weights[1]), 2: float(weights[2])}
        print(f"  Class weights (manual override): {resolved}")
        return resolved

    ns["compute_class_weights_from_y"] = _manual_resolver
    experiment_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fold_spec in fold_plan:
            repeat_idx = fold_spec["repeat"]
            fold_idx = fold_spec["fold"]
            global_fold_idx = fold_spec["global_fold"]
            repeat_seed = fold_spec["cv_seed"]
            train_idx = fold_spec["train_idx"]
            val_idx = fold_spec["val_idx"]

            print("\n" + "=" * 68)
            print(
                f"{experiment_name} | Repeat {repeat_idx}/3 | Fold {fold_idx}/5 | "
                f"Run {global_fold_idx}/{total_fold_runs} | seed={repeat_seed}"
            )
            print("=" * 68)

            is_reliable, val_stats = ns["analyze_validation_set"](
                metadata,
                val_idx,
                min_severe_records=effective_min_severe,
            )
            print(f"  Reliable fold: {is_reliable} | Severe step-units: {val_stats['n_severe_step_units']}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            g_train, g_val = groups[train_idx], groups[val_idx]
            meta_val = metadata.iloc[val_idx].reset_index(drop=True)

            clin_train = clin_all[train_idx] if clin_all is not None else None
            clin_val = clin_all[val_idx] if clin_all is not None else None

            train_recs = set(g_train)
            val_recs = set(g_val)
            overlap = train_recs & val_recs
            if overlap:
                raise RuntimeError(f"Data leakage detected across folds: {sorted(overlap)[:5]}")

            norm_stats = ns["compute_norm_stats"](X_train)
            X_train_n = ns["apply_norm"](X_train, norm_stats)
            X_val_n = ns["apply_norm"](X_val, norm_stats)

            if clin_train is not None:
                clin_train_n, _ = ns["normalize_clinical_features"](clin_train, clin_train)
                clin_val_n, _ = ns["normalize_clinical_features"](clin_train, clin_val)
            else:
                clin_train_n = None
                clin_val_n = None

            tf.keras.backend.clear_session()
            model = ns["build_model"](
                input_length=cfg["segment_length"],
                num_classes=ns["NUM_CLASSES"],
                temporal_filters=cfg["temporal_filters"],
                temporal_kernel=cfg["temporal_kernel"],
                separable_filters=cfg["separable_filters"],
                separable_kernels=cfg["separable_kernels"],
                projection_filters=cfg["projection_filters"],
                dropout_rate=cfg["dropout_rate"],
                n_clinical_features=ns["N_CLINICAL_FEATURES"] if cfg.get("use_clinical_features", True) else 0,
            )

            ns["train_from_scratch"](
                model,
                X_train_n,
                y_train,
                X_val_n,
                y_val,
                meta_val,
                cfg,
                save_path=str(experiment_dir / f"repeat{repeat_idx}_fold{fold_idx}_best.weights.h5"),
                clin_train=clin_train_n,
                clin_val=clin_val_n,
            )

            y_proba_val = model.predict(ns["_make_model_input"](X_val_n, clin_val_n), batch_size=32, verbose=0)
            _, step_y_true, step_y_pred = ns["aggregate_to_record_step_level"](y_proba_val, meta_val)
            metrics = ns["compute_metrics"](step_y_true, step_y_pred)
            metrics["repeat"] = repeat_idx
            metrics["fold"] = fold_idx
            metrics["global_fold"] = global_fold_idx
            metrics["cv_seed"] = repeat_seed
            metrics["is_reliable"] = bool(is_reliable)
            metrics["n_train_records"] = len(train_recs)
            metrics["n_val_records"] = len(val_recs)
            metrics["n_val_step_units"] = val_stats["n_val_step_units"]
            metrics["n_severe_val_step_units"] = val_stats["n_severe_step_units"]
            metrics["requested_min_severe_threshold"] = requested_min_severe
            metrics["effective_min_severe_threshold"] = effective_min_severe
            fold_results.append(metrics)

            all_step_y_true.extend(step_y_true)
            all_step_y_pred.extend(step_y_pred)
            ns["print_metrics"](metrics, prefix=f"Repeat {repeat_idx} Fold {fold_idx} ")

            del model
            del y_proba_val
            del X_train, X_val, y_train, y_val, meta_val
            del g_train, g_val, train_recs, val_recs
            del X_train_n, X_val_n, clin_train, clin_val, clin_train_n, clin_val_n
            gc.collect()
            tf.keras.backend.clear_session()
    finally:
        ns["compute_class_weights_from_y"] = original_resolver

    overall_cm = ns["confusion_matrix"](all_step_y_true, all_step_y_pred, labels=list(range(ns["NUM_CLASSES"])))
    ns["plot_confusion_matrix"](
        overall_cm,
        title=f"Overall Record-Step Confusion Matrix ({experiment_name})",
        save_path=str(experiment_dir / "overall_cm.png"),
    )
    ns["plot_fold_summary"](fold_results, save_path=str(experiment_dir / "fold_summary.png"))

    summary = summarize_results(fold_results, weights, cfg)
    save_json(summary_path, summary)
    print(f"\nSaved {experiment_name} summary to {summary_path}")
    return summary_path


def aggregate_results() -> Path:
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Baseline summary not found: {BASELINE_SUMMARY_PATH}")

    with open(BASELINE_SUMMARY_PATH, "r") as baseline_file:
        baseline = json.load(baseline_file)

    result_rows = []
    for weights in WEIGHT_GRID:
        result_path = OUTPUT_ROOT / weights_to_name(weights) / "results.json"
        if not result_path.exists():
            continue
        with open(result_path, "r") as result_file:
            result = json.load(result_file)
        result_rows.append(
            {
                "name": weights_to_name(weights),
                "manual_class_weights": result["manual_class_weights"],
                "mean_balanced_acc": result["mean_balanced_acc"],
                "mean_macro_f1": result["mean_macro_f1"],
                "mean_recall_Normal": result["mean_recall_Normal"],
                "mean_recall_Mild": result["mean_recall_Mild"],
                "mean_recall_Severe": result["mean_recall_Severe"],
                "passes_guardrails": result["passes_guardrails"],
                "delta_balanced_acc_vs_baseline": float(result["mean_balanced_acc"] - baseline["mean_balanced_acc"]),
            }
        )

    result_rows.sort(key=lambda row: row["mean_balanced_acc"], reverse=True)
    aggregate = {
        "baseline": {
            "manual_class_weights": baseline.get("manual_class_weights"),
            "mean_balanced_acc": baseline["mean_balanced_acc"],
            "mean_macro_f1": baseline["mean_macro_f1"],
            "mean_recall_Normal": baseline["mean_recall_Normal"],
            "mean_recall_Mild": baseline["mean_recall_Mild"],
            "mean_recall_Severe": baseline["mean_recall_Severe"],
        },
        "ranking": result_rows,
    }
    output_path = OUTPUT_ROOT / "class_weight_sweep_summary.json"
    save_json(output_path, aggregate)
    print(f"Saved sweep aggregate summary to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs=3, type=float, metavar=("NORMAL", "MILD", "SEVERE"))
    parser.add_argument("--all", action="store_true", help="Run the full predefined grid in fresh subprocesses.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild the aggregate ranking JSON.")
    parser.add_argument("--force", action="store_true", help="Re-run even if a result file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    if args.all:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for weights in WEIGHT_GRID:
            result_path = OUTPUT_ROOT / weights_to_name(weights) / "results.json"
            if result_path.exists() and not args.force:
                print(f"Skipping {weights_to_name(weights)} because results already exist.")
                continue
            cmd = [
                sys.executable,
                str(Path(__file__)),
                "--weights",
                str(weights[0]),
                str(weights[1]),
                str(weights[2]),
            ]
            if args.force:
                cmd.append("--force")
            print(f"\nLaunching fresh process for weights={weights}")
            subprocess.run(cmd, check=True)
        aggregate_results()
        return

    if args.weights is None:
        raise SystemExit("Provide --weights NORMAL MILD SEVERE or use --all.")

    run_single_experiment((float(args.weights[0]), float(args.weights[1]), float(args.weights[2])), force=args.force)


if __name__ == "__main__":
    main()