import json
import os
import sys
import importlib
import gc
from pathlib import Path

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


def _load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as notebook_file:
        return json.load(notebook_file)



def _execute_setup_cells(notebook):
    namespace = {}

    import tensorflow as tf
    tf.keras.backend.clear_session()

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        source = "".join(cell.source)

        # Skip empty cells
        if not source.strip():
            continue

        code = compile(source, f"{NOTEBOOK_PATH.name}:cell_{index}", "exec")

        # Force setup-time execution onto CPU
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



def run() -> dict:
    os.chdir(SCRIPT_DIR)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    helper_module_name = "tenfold_tuning_helpers"
    if helper_module_name in sys.modules:
        helper_module = importlib.reload(sys.modules[helper_module_name])
    else:
        helper_module = importlib.import_module(helper_module_name)
    run_tenfold_label_smoothing_sweep = helper_module.run_tenfold_label_smoothing_sweep

    notebook = _load_notebook(NOTEBOOK_PATH)
    namespace = _execute_setup_cells(notebook)
    cfg_base = dict(namespace["CFG"])
    cfg_base["scratch_batch"] = min(int(cfg_base.get("scratch_batch", 8)), 2)
    cfg_base["scratch_eval_batch"] = min(int(cfg_base.get("scratch_eval_batch", 32)), 4)

    class_weight_grid = [
        {0: 0.9, 1: 1.1, 2: 2.3},
        {0: 0.9, 1: 1.2, 2: 2.3},
        {0: 0.9, 1: 1.3, 2: 2.3},
        {0: 0.9, 1: 1.2, 2: 2.5},
        {0: 0.9, 1: 1.3, 2: 2.5},
        {0: 1.0, 1: 1.2, 2: 2.3},
        {0: 1.0, 1: 1.2, 2: 2.5},
    ]
    label_smoothing_grid = [0.0, 0.02, 0.05]
    calibration_boost_grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

    original_compute_class_weights = namespace["compute_class_weights_from_y"]
    combined_summary_frames = []
    combined_summaries = []
    best_run = None

    try:
        for class_weights in class_weight_grid:
            weight_slug = _class_weights_to_slug(class_weights)
            weight_output_dir = OUTPUT_DIR / weight_slug
            namespace["compute_class_weights_from_y"] = _manual_class_weight_resolver_factory(
                namespace["NUM_CLASSES"],
                class_weights,
            )

            summary_df, summaries, weight_best_run = run_tenfold_label_smoothing_sweep(
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
                label_smoothing_grid=label_smoothing_grid,
                calibration_boost_grid=calibration_boost_grid,
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

            gc.collect()
            namespace["tf"].keras.backend.clear_session()
    finally:
        namespace["compute_class_weights_from_y"] = original_compute_class_weights

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
    summary_df.to_csv(OUTPUT_DIR / "class_weight_label_smoothing_sweep_summary.csv", index=False)
    with (OUTPUT_DIR / "class_weight_label_smoothing_sweep_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(combined_summaries, summary_file, indent=2)

    namespace["best_current_weights"] = dict(best_run["manual_class_weights"])
    namespace["tenfold_cfg"] = cfg_base
    namespace["class_weight_grid"] = class_weight_grid
    namespace["label_smoothing_grid"] = label_smoothing_grid
    namespace["calibration_boost_grid"] = calibration_boost_grid
    namespace["tenfold_tuning_output_dir"] = OUTPUT_DIR
    namespace["tenfold_tuning_summary_df"] = summary_df
    namespace["tenfold_tuning_summaries"] = combined_summaries
    namespace["best_tenfold_run"] = best_run
    return namespace


if __name__ == "__main__":
    result = run()
    print("\nBest run:")
    print(result["best_tenfold_run"])
