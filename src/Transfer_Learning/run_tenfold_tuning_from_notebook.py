import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "ten_min_segment_pipeline_v10.4.ipynb"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "step23_softmax_tenfold_tuning"
STOP_MARKERS = [
    "Running fixed class-weight experiment",
    "STEP 2 + STEP 3 SOFTMAX WITH FIXED CLASS WEIGHTS",
    "FINAL SUMMARY",
]
TRUNCATE_CELL_MARKERS = [
    "model_test = build_model(",
]
CLASS_WEIGHT_GRID = [
    {0: 0.9, 1: 1.1, 2: 2.3},
    {0: 0.9, 1: 1.2, 2: 2.3},
    {0: 0.9, 1: 1.2, 2: 2.5},
    {0: 0.9, 1: 1.3, 2: 2.5},
    {0: 1.0, 1: 1.2, 2: 2.5},
]
LABEL_SMOOTHING_GRID = [0.0, 0.02, 0.05]
CALIBRATION_BOOST_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]


def _load_notebook(path: Path):
    return nbformat.read(path, as_version=4)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_child_result(result_path: Path, summary_df: pd.DataFrame, summaries: list, best_run: dict, cfg_base: dict) -> None:
    payload = {
        "summary_df": summary_df.to_dict(orient="records"),
        "summaries": summaries,
        "best_run": best_run,
        "cfg_base": cfg_base,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(_jsonify(payload), result_file, indent=2)


def _load_child_result(result_path: Path) -> dict:
    with result_path.open("r", encoding="utf-8") as result_file:
        payload = json.load(result_file)
    payload["summary_df"] = pd.DataFrame(payload["summary_df"])
    return payload


def _strip_setup_side_effects(source: str) -> str:
    for marker in TRUNCATE_CELL_MARKERS:
        marker_index = source.find(marker)
        if marker_index != -1:
            return source[:marker_index].rstrip()
    return source


def _execute_setup_cells(notebook):
    namespace = {"__name__": "__main__"}
    cells = notebook.cells if hasattr(notebook, "cells") else notebook["cells"]

    import tensorflow as tf

    tf.keras.backend.clear_session()

    for index, cell in enumerate(cells):
        cell_type = cell.cell_type if hasattr(cell, "cell_type") else cell.get("cell_type")
        if cell_type != "code":
            continue

        source = cell.source if hasattr(cell, "source") else cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if any(marker in source for marker in STOP_MARKERS):
            print(f"Stopping notebook setup at cell {index} due to stop marker.")
            break

        source = _strip_setup_side_effects(source)

        if not source.strip():
            continue

        code = compile(source, f"{NOTEBOOK_PATH.name}:cell_{index}", "exec")

        with tf.device("/CPU:0"):
            exec(code, namespace)

    return namespace


def _manual_class_weight_resolver_factory(num_classes: int, manual_weights: dict):
    def _resolver(y_train):
        resolved = {
            int(class_idx): float(manual_weights.get(class_idx, 1.0))
            for class_idx in range(num_classes)
        }
        print(f"  Class weights (manual override): {resolved}")
        return resolved

    return _resolver


def _class_weights_to_slug(class_weights: dict) -> str:
    return "cw_" + "_".join(
        str(float(class_weights[class_idx])).replace(".", "p")
        for class_idx in sorted(class_weights)
    )


def _build_namespace_and_cfg():
    os.chdir(SCRIPT_DIR)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    helper_module_name = "tenfold_tuning_helpers"
    if helper_module_name in sys.modules:
        helper_module = importlib.reload(sys.modules[helper_module_name])
    else:
        helper_module = importlib.import_module(helper_module_name)

    notebook = _load_notebook(NOTEBOOK_PATH)
    namespace = _execute_setup_cells(notebook)
    cfg_base = dict(namespace["CFG"])
    cfg_base["scratch_batch"] = min(int(cfg_base.get("scratch_batch", 8)), 2)
    cfg_base["scratch_eval_batch"] = min(int(cfg_base.get("scratch_eval_batch", 32)), 4)
    return helper_module, namespace, cfg_base


def _run_single_experiment(class_weights: dict, label_smoothing: float) -> dict:
    helper_module, namespace, cfg_base = _build_namespace_and_cfg()
    run_tenfold_label_smoothing_sweep = helper_module.run_tenfold_label_smoothing_sweep

    original_compute_class_weights = namespace["compute_class_weights_from_y"]
    weight_output_dir = OUTPUT_DIR / _class_weights_to_slug(class_weights)

    try:
        namespace["compute_class_weights_from_y"] = _manual_class_weight_resolver_factory(
            namespace["NUM_CLASSES"],
            class_weights,
        )
        summary_df, summaries, best_run = run_tenfold_label_smoothing_sweep(
            X_all=namespace["X_all"],
            y_all=namespace["y_all"],
            groups_all=namespace["groups_all"],
            meta_all=namespace["meta_all"],
            cfg_base=cfg_base,
            output_root=weight_output_dir,
            x_raw=namespace["X_all"],
            feature_missing_frac=namespace["meta_all"]["missing_frac_prefill"].to_numpy(dtype=np.float32),
            class_names=namespace["CLASS_NAMES"],
            num_classes=namespace["NUM_CLASSES"],
            sampling_rate=namespace["SAMPLING_RATE"],
            n_clinical_features=namespace["N_CLINICAL_FEATURES"],
            best_current_weights=class_weights,
            label_smoothing_grid=[float(label_smoothing)],
            calibration_boost_grid=CALIBRATION_BOOST_GRID,
            tf=namespace["tf"],
            ensure_step_keys=namespace["_ensure_step_keys"],
            build_step_target_table=namespace["build_step_target_table"],
            stratified_record_kfold_with_constraint=namespace["stratified_record_kfold_with_constraint"],
            analyze_validation_set=namespace["analyze_validation_set"],
            compute_clinical_features=namespace["compute_clinical_features"],
            compute_norm_stats=namespace["compute_norm_stats"],
            apply_norm=namespace["apply_norm"],
            normalize_clinical_features=namespace["normalize_clinical_features"],
            build_model=namespace["build_model"],
            train_from_scratch=namespace["train_from_scratch"],
            make_model_input=namespace["_make_model_input"],
            compute_metrics=namespace["compute_metrics"],
            plot_confusion_matrix=namespace["plot_confusion_matrix"],
            plot_fold_summary=namespace["plot_fold_summary"],
        )
        return {
            "summary_df": summary_df,
            "summaries": summaries,
            "best_run": best_run,
            "cfg_base": cfg_base,
        }
    finally:
        namespace["compute_class_weights_from_y"] = original_compute_class_weights
        gc.collect()
        namespace["tf"].keras.backend.clear_session()


def _run_single_experiment_in_child(class_weights: dict, label_smoothing: float, result_path: Path) -> None:
    result = _run_single_experiment(class_weights, label_smoothing)
    _serialize_child_result(
        result_path,
        result["summary_df"],
        result["summaries"],
        result["best_run"],
        result["cfg_base"],
    )


def _launch_child_experiment(class_weights: dict, label_smoothing: float, result_path: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        str(result_path),
        "--weights",
        str(class_weights[0]),
        str(class_weights[1]),
        str(class_weights[2]),
        "--label-smoothing",
        str(float(label_smoothing)),
    ]
    completed = subprocess.run(command, check=False, cwd=str(SCRIPT_DIR))
    if completed.returncode != 0:
        raise RuntimeError(
            "Single experiment child process failed for "
            f"weights={class_weights}, label_smoothing={label_smoothing}."
        )
    try:
        return _load_child_result(result_path)
    finally:
        if result_path.exists():
            result_path.unlink()


def run() -> dict:
    combined_summary_frames = []
    combined_summaries = []
    best_run = None
    cfg_base = None
    child_result_dir = OUTPUT_DIR / "_child_runs"

    for class_weights in CLASS_WEIGHT_GRID:
        for label_smoothing in LABEL_SMOOTHING_GRID:
            weight_slug = _class_weights_to_slug(class_weights)
            result_slug = f"{weight_slug}__ls_{float(label_smoothing):.2f}".replace(".", "p")
            result_path = child_result_dir / f"{result_slug}.json"
            child_result = _launch_child_experiment(class_weights, label_smoothing, result_path)

            if cfg_base is None:
                cfg_base = dict(child_result["cfg_base"])

            summary_df = child_result["summary_df"]
            summaries = child_result["summaries"]
            weight_best_run = child_result["best_run"]

            combined_summary_frames.append(summary_df)
            combined_summaries.extend(summaries)

            if best_run is None:
                best_run = dict(weight_best_run)
            else:
                current_rank = (
                    float(weight_best_run["calibrated_balanced_accuracy"]),
                    float(weight_best_run["calibrated_macro_f1"]),
                    float(weight_best_run["raw_balanced_accuracy"]),
                    str(weight_best_run["class_weight_slug"]),
                    float(weight_best_run["label_smoothing"]),
                )
                best_rank = (
                    float(best_run["calibrated_balanced_accuracy"]),
                    float(best_run["calibrated_macro_f1"]),
                    float(best_run["raw_balanced_accuracy"]),
                    str(best_run["class_weight_slug"]),
                    float(best_run["label_smoothing"]),
                )
                if current_rank > best_rank:
                    best_run = dict(weight_best_run)

    summary_df = pd.concat(combined_summary_frames, ignore_index=True).sort_values(
        [
            "calibrated_balanced_accuracy",
            "calibrated_macro_f1",
            "raw_balanced_accuracy",
            "class_weight_slug",
            "label_smoothing",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "class_weight_label_smoothing_sweep_summary.csv", index=False)
    with (OUTPUT_DIR / "class_weight_label_smoothing_sweep_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(combined_summaries, summary_file, indent=2)

    return {
        "best_current_weights": dict(best_run["manual_class_weights"]),
        "tenfold_cfg": cfg_base or {},
        "class_weight_grid": CLASS_WEIGHT_GRID,
        "label_smoothing_grid": LABEL_SMOOTHING_GRID,
        "calibration_boost_grid": CALIBRATION_BOOST_GRID,
        "tenfold_tuning_output_dir": OUTPUT_DIR,
        "tenfold_tuning_summary_df": summary_df,
        "tenfold_tuning_summaries": combined_summaries,
        "best_tenfold_run": best_run,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run", type=str)
    parser.add_argument("--weights", nargs=3, type=float, metavar=("NORMAL", "MILD", "SEVERE"))
    parser.add_argument("--label-smoothing", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.child_run:
        if args.weights is None or args.label_smoothing is None:
            raise SystemExit("Child mode requires --weights NORMAL MILD SEVERE and --label-smoothing.")
        class_weights = {
            0: float(args.weights[0]),
            1: float(args.weights[1]),
            2: float(args.weights[2]),
        }
        _run_single_experiment_in_child(class_weights, float(args.label_smoothing), Path(args.child_run))
    else:
        result = run()
        print("\nBest run:")
        print(result["best_tenfold_run"])
