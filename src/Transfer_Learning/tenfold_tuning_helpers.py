import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reset_tf_state(tf):
    tf.keras.backend.clear_session()
    gc.collect()


def class_weights_to_slug(class_weights):
    return "cw_" + "_".join(str(float(class_weights[class_idx])).replace(".", "p") for class_idx in sorted(class_weights))


def build_step_probability_table(y_proba, metadata, probability_columns, ensure_step_keys):
    meta = ensure_step_keys(metadata.reset_index(drop=True))
    rows = []

    for step_key in meta["step_key"].unique():
        mask = meta["step_key"] == step_key
        mean_proba = y_proba[mask].mean(axis=0)
        true_label = int(meta.loc[mask, "label"].iloc[0])
        pred_label = int(np.argmax(mean_proba))
        rows.append(
            {
                "step_key": step_key,
                "rec_id": str(meta.loc[mask, "rec_id"].iloc[0]),
                "step": int(meta.loc[mask, "step"].iloc[0]),
                "true_label": true_label,
                "pred_label": pred_label,
                **{col: float(value) for col, value in zip(probability_columns, mean_proba)},
            }
        )

    return pd.DataFrame(rows)


def apply_step_level_severe_boost(step_probability_table, severe_boost, probability_columns):
    calibrated = step_probability_table.copy()
    prob_values = calibrated[probability_columns].to_numpy(dtype=np.float64)

    if severe_boost > 0:
        prob_values[:, 2] += severe_boost
        prob_values = prob_values / prob_values.sum(axis=1, keepdims=True)

    calibrated.loc[:, probability_columns] = prob_values
    calibrated["pred_label"] = np.argmax(prob_values, axis=1).astype(int)
    calibrated["severe_boost"] = float(severe_boost)
    return calibrated


def compute_step_probability_metrics(step_probability_table, compute_metrics):
    metrics = compute_metrics(
        step_probability_table["true_label"].to_numpy(dtype=int),
        step_probability_table["pred_label"].to_numpy(dtype=int),
    )
    metrics["n_record_steps"] = int(len(step_probability_table))
    return metrics


def metrics_to_jsonable(metrics):
    jsonable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            jsonable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            jsonable[key] = value.item()
        else:
            jsonable[key] = value
    return jsonable


def summarize_fold_metrics(fold_results, class_names):
    summary = {
        "n_folds": int(len(fold_results)),
        "n_reliable_folds": int(sum(result.get("is_reliable", True) for result in fold_results)),
    }
    for metric_name in ["accuracy", "balanced_accuracy", "macro_f1"]:
        values = np.array([result[metric_name] for result in fold_results], dtype=np.float64)
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())
    for class_name in class_names:
        for metric_name in ["precision", "recall", "f1"]:
            key = f"{metric_name}_{class_name}"
            values = np.array([result[key] for result in fold_results], dtype=np.float64)
            summary[f"mean_{key}"] = float(values.mean())
            summary[f"std_{key}"] = float(values.std())
    return summary


def search_optimal_oof_severe_boost(step_probability_table, boost_grid, probability_columns, compute_metrics):
    search_rows = []
    best_choice = None

    for severe_boost in boost_grid:
        calibrated = apply_step_level_severe_boost(step_probability_table, severe_boost, probability_columns)
        metrics = compute_step_probability_metrics(calibrated, compute_metrics)
        row = {
            "severe_boost": float(severe_boost),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "recall_Normal": float(metrics["recall_Normal"]),
            "recall_Mild": float(metrics["recall_Mild"]),
            "recall_Severe": float(metrics["recall_Severe"]),
        }
        search_rows.append(row)

        candidate_rank = (
            row["balanced_accuracy"],
            row["macro_f1"],
            row["accuracy"],
            -row["severe_boost"],
        )
        if best_choice is None or candidate_rank > best_choice[0]:
            best_choice = (candidate_rank, float(severe_boost), metrics)

    search_df = pd.DataFrame(search_rows).sort_values(
        ["balanced_accuracy", "macro_f1", "accuracy", "severe_boost"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return best_choice[1], best_choice[2], search_df


def build_batch_size_candidates(initial_batch_size):
    initial = max(int(initial_batch_size), 1)
    candidates = []
    current = initial
    while current >= 1:
        if current not in candidates:
            candidates.append(current)
        if current == 1:
            break
        current = max(current // 2, 1)
    return candidates


def is_resource_exhaustion_error(error, tf):
    if isinstance(error, (tf.errors.ResourceExhaustedError, MemoryError)):
        return True
    message = str(error).lower()
    return "resourceexhausted" in message or "resource exhausted" in message or "oom" in message


def run_grouped_cv_with_oof(
    *,
    X,
    y,
    groups,
    metadata,
    cfg,
    output_dir,
    X_raw,
    feature_missing_frac,
    class_names,
    num_classes,
    sampling_rate,
    n_clinical_features,
    tf,
    ensure_step_keys,
    stratified_record_kfold_with_constraint,
    analyze_validation_set,
    compute_clinical_features,
    compute_norm_stats,
    apply_norm,
    normalize_clinical_features,
    build_model,
    train_from_scratch,
    make_model_input,
    compute_metrics,
    plot_confusion_matrix,
    plot_fold_summary,
):
    if len(X) != len(y) or len(X) != len(groups) or len(X) != len(metadata):
        raise ValueError("X, y, groups, and metadata must all have the same length")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = ensure_step_keys(metadata.reset_index(drop=True))
    probability_columns = [f"prob_{class_name.lower()}" for class_name in class_names]

    n_splits = int(cfg["n_splits"])
    use_clinical = bool(cfg.get("use_clinical_features", True))
    requested_min_severe = int(cfg.get("min_severe_records", 10))

    X_for_features = X_raw if X_raw is not None else X
    if use_clinical:
        print("Computing engineered features from cleaned FHR windows...")
        clin_all = compute_clinical_features(
            X_for_features,
            fs=sampling_rate,
            missing_frac_prefill=feature_missing_frac,
        )
        print(f"  Engineered feature matrix: {clin_all.shape}")
    else:
        clin_all = None

    print(f"\nGenerating {n_splits}-fold CV with requested >= {requested_min_severe} Severe step-units/fold...")
    folds, effective_min_severe = stratified_record_kfold_with_constraint(
        metadata,
        n_splits=n_splits,
        min_severe_records=requested_min_severe,
        max_attempts=cfg.get("max_cv_attempts", 100),
        cv_seed=cfg["cv_seed"],
    )
    if effective_min_severe != requested_min_severe:
        print(f"  Effective reliability threshold: {effective_min_severe} Severe step-units/fold")

    fold_results = []
    oof_step_tables = []
    initial_train_batch_size = int(cfg.get("scratch_batch", 8))
    initial_eval_batch_size = int(cfg.get("scratch_eval_batch", 32))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1} / {n_splits}")
        print(f"{'=' * 60}")

        is_reliable, val_stats = analyze_validation_set(
            metadata,
            val_idx,
            min_severe_records=effective_min_severe,
        )

        print(f"  Validation quality: {'RELIABLE' if is_reliable else 'UNRELIABLE'}")
        for class_id, class_name in enumerate(class_names):
            print(
                f"    {class_name}: {val_stats['step_units_per_class'][class_id]} step-units, "
                f"{val_stats['windows_per_class'][class_id]} windows"
            )

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_train, g_val = groups[train_idx], groups[val_idx]
        meta_val = metadata.iloc[val_idx].reset_index(drop=True)

        train_recs = set(g_train)
        val_recs = set(g_val)
        overlap = train_recs & val_recs
        assert len(overlap) == 0, f"DATA LEAKAGE! {len(overlap)} records in both sets: {overlap}"

        clin_train = clin_all[train_idx] if clin_all is not None else None
        clin_val = clin_all[val_idx] if clin_all is not None else None

        norm_stats = compute_norm_stats(X_train)
        X_train_n = apply_norm(X_train, norm_stats)
        X_val_n = apply_norm(X_val, norm_stats)

        if clin_train is not None:
            clin_train_n, _ = normalize_clinical_features(clin_train, clin_train)
            clin_val_n, _ = normalize_clinical_features(clin_train, clin_val)
        else:
            clin_train_n = None
            clin_val_n = None

        save_path = str(output_dir / f"fold{fold_idx + 1}_best.weights.h5")
        batch_size_candidates = build_batch_size_candidates(initial_train_batch_size)
        last_training_error = None
        used_train_batch_size = None
        used_eval_batch_size = None
        y_proba_val = None
        model = None

        for train_batch_size in batch_size_candidates:
            eval_batch_size = max(1, min(initial_eval_batch_size, train_batch_size * 2))
            cfg_attempt = dict(cfg)
            cfg_attempt["scratch_batch"] = int(train_batch_size)
            cfg_attempt["scratch_eval_batch"] = int(eval_batch_size)

            reset_tf_state(tf)

            n_clin = n_clinical_features if use_clinical else 0
            model = build_model(
                input_length=cfg_attempt["segment_length"],
                num_classes=num_classes,
                temporal_filters=cfg_attempt["temporal_filters"],
                temporal_kernel=cfg_attempt["temporal_kernel"],
                separable_filters=cfg_attempt["separable_filters"],
                separable_kernels=cfg_attempt["separable_kernels"],
                projection_filters=cfg_attempt["projection_filters"],
                dropout_rate=cfg_attempt["dropout_rate"],
                n_clinical_features=n_clin,
            )

            try:
                if train_batch_size != initial_train_batch_size:
                    print(
                        f"  Retrying fold {fold_idx + 1} with scratch_batch={train_batch_size} "
                        f"and scratch_eval_batch={eval_batch_size}"
                    )

                train_from_scratch(
                    model,
                    X_train_n,
                    y_train,
                    X_val_n,
                    y_val,
                    meta_val,
                    cfg_attempt,
                    save_path=save_path,
                    clin_train=clin_train_n,
                    clin_val=clin_val_n,
                )

                y_proba_val = model.predict(
                    make_model_input(X_val_n, clin_val_n),
                    batch_size=eval_batch_size,
                    verbose=0,
                )
                used_train_batch_size = int(train_batch_size)
                used_eval_batch_size = int(eval_batch_size)
                break
            except Exception as error:
                if not is_resource_exhaustion_error(error, tf):
                    raise
                last_training_error = error
                print(
                    f"  Resource exhaustion for fold {fold_idx + 1} at scratch_batch={train_batch_size}."
                )
                model = None
                reset_tf_state(tf)

        if y_proba_val is None:
            raise last_training_error

        step_table = build_step_probability_table(y_proba_val, meta_val, probability_columns, ensure_step_keys)
        step_table["fold"] = int(fold_idx + 1)
        step_table["is_reliable"] = bool(is_reliable)
        step_table["requested_min_severe_threshold"] = int(requested_min_severe)
        step_table["effective_min_severe_threshold"] = int(effective_min_severe)
        step_table["used_train_batch_size"] = int(used_train_batch_size)
        step_table["used_eval_batch_size"] = int(used_eval_batch_size)
        oof_step_tables.append(step_table)

        metrics = compute_step_probability_metrics(step_table, compute_metrics)
        metrics["fold"] = int(fold_idx + 1)
        metrics["n_train_records"] = int(len(train_recs))
        metrics["n_val_records"] = int(len(val_recs))
        metrics["n_val_step_units"] = int(val_stats["n_val_step_units"])
        metrics["is_reliable"] = bool(is_reliable)
        metrics["n_severe_val_step_units"] = int(val_stats["n_severe_step_units"])
        metrics["requested_min_severe_threshold"] = int(requested_min_severe)
        metrics["effective_min_severe_threshold"] = int(effective_min_severe)
        metrics["used_train_batch_size"] = int(used_train_batch_size)
        metrics["used_eval_batch_size"] = int(used_eval_batch_size)
        fold_results.append(metrics)

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=(f"Fold {fold_idx + 1} — Record-Step CM" + (" [UNRELIABLE]" if not is_reliable else "")),
            save_path=str(output_dir / f"fold{fold_idx + 1}_cm.png"),
        )
        plt.close("all")
        del model
        del y_proba_val
        del X_train_n
        del X_val_n
        del y_train
        del y_val
        del clin_train
        del clin_val
        del clin_train_n
        del clin_val_n
        reset_tf_state(tf)

    oof_step_probabilities = pd.concat(oof_step_tables, ignore_index=True)
    raw_oof_metrics = compute_step_probability_metrics(oof_step_probabilities, compute_metrics)
    plot_confusion_matrix(
        raw_oof_metrics["confusion_matrix"],
        title="Overall OOF Record-Step Confusion Matrix",
        save_path=str(output_dir / "overall_oof_cm.png"),
    )
    plt.close("all")
    plot_fold_summary(fold_results, save_path=str(output_dir / "fold_summary.png"))
    plt.close("all")

    return {
        "fold_results": fold_results,
        "fold_summary": summarize_fold_metrics(fold_results, class_names),
        "oof_step_probabilities": oof_step_probabilities,
        "oof_metrics": raw_oof_metrics,
        "probability_columns": probability_columns,
    }


def run_tenfold_label_smoothing_sweep(
    *,
    X_all,
    y_all,
    groups_all,
    meta_all,
    cfg_base,
    output_root,
    x_raw,
    feature_missing_frac,
    class_names,
    num_classes,
    sampling_rate,
    n_clinical_features,
    best_current_weights,
    label_smoothing_grid,
    calibration_boost_grid,
    tf,
    ensure_step_keys,
    build_step_target_table,
    stratified_record_kfold_with_constraint,
    analyze_validation_set,
    compute_clinical_features,
    compute_norm_stats,
    apply_norm,
    normalize_clinical_features,
    build_model,
    train_from_scratch,
    make_model_input,
    compute_metrics,
    plot_confusion_matrix,
    plot_fold_summary,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"prob_{class_name.lower()}" for class_name in class_names]

    tenfold_tuning_runs = []
    tenfold_tuning_summaries = []
    step_unit_table = build_step_target_table(meta_all)

    print("\n" + "#" * 78)
    print("# EXPERIMENT A: Tenfold Label-Smoothing Sweep + Pooled OOF Calibration")
    print("#" * 78)
    print(
        f"\nDataset: {len(X_all)} windows from {len(step_unit_table)} record-steps across "
        f"{len(np.unique(groups_all))} records"
    )
    print(f"Window shape: {X_all.shape}")
    if len(meta_all) > 0:
        print(f"Steps: {meta_all['step'].value_counts().sort_index().to_dict()}")
    print(f"Classes: {dict(zip(class_names, [int(np.sum(y_all == i)) for i in range(num_classes)]))}")
    print(f"Manual class weights: {best_current_weights}")

    for label_smoothing in label_smoothing_grid:
        run_slug = f"ls_{label_smoothing:.2f}".replace(".", "p")
        run_output_dir = output_root / run_slug
        run_output_dir.mkdir(parents=True, exist_ok=True)
        reset_tf_state(tf)

        cfg_run = dict(cfg_base)
        cfg_run["n_splits"] = 10
        cfg_run["label_smoothing"] = float(label_smoothing)
        cfg_run["severe_boost"] = 0.0
        cfg_run["use_class_weights"] = True

        print("\n" + "=" * 78)
        print(f"Running tenfold CV for label_smoothing={label_smoothing:.2f}")
        print(f"Output directory: {run_output_dir}")
        print("=" * 78)

        cv_result = run_grouped_cv_with_oof(
            X=X_all,
            y=y_all,
            groups=groups_all,
            metadata=meta_all,
            cfg=cfg_run,
            output_dir=run_output_dir,
            X_raw=x_raw,
            feature_missing_frac=feature_missing_frac,
            class_names=class_names,
            num_classes=num_classes,
            sampling_rate=sampling_rate,
            n_clinical_features=n_clinical_features,
            tf=tf,
            ensure_step_keys=ensure_step_keys,
            stratified_record_kfold_with_constraint=stratified_record_kfold_with_constraint,
            analyze_validation_set=analyze_validation_set,
            compute_clinical_features=compute_clinical_features,
            compute_norm_stats=compute_norm_stats,
            apply_norm=apply_norm,
            normalize_clinical_features=normalize_clinical_features,
            build_model=build_model,
            train_from_scratch=train_from_scratch,
            make_model_input=make_model_input,
            compute_metrics=compute_metrics,
            plot_confusion_matrix=plot_confusion_matrix,
            plot_fold_summary=plot_fold_summary,
        )

        raw_oof_table = cv_result["oof_step_probabilities"].copy()
        raw_metrics = compute_step_probability_metrics(raw_oof_table, compute_metrics)
        best_boost, calibrated_metrics, calibration_search_df = search_optimal_oof_severe_boost(
            raw_oof_table,
            calibration_boost_grid,
            probability_columns,
            compute_metrics,
        )
        calibrated_oof_table = apply_step_level_severe_boost(raw_oof_table, best_boost, probability_columns)

        raw_oof_table.to_csv(run_output_dir / "oof_step_probabilities.csv", index=False)
        calibrated_oof_table.to_csv(run_output_dir / "oof_step_probabilities_calibrated.csv", index=False)
        calibration_search_df.to_csv(run_output_dir / "oof_calibration_grid.csv", index=False)

        plot_confusion_matrix(
            raw_metrics["confusion_matrix"],
            title=f"OOF Record-Step CM | label_smoothing={label_smoothing:.2f} | raw",
            save_path=str(run_output_dir / "oof_confusion_matrix_raw.png"),
        )
        plt.close("all")
        plot_confusion_matrix(
            calibrated_metrics["confusion_matrix"],
            title=f"OOF Record-Step CM | label_smoothing={label_smoothing:.2f} | calibrated",
            save_path=str(run_output_dir / "oof_confusion_matrix_calibrated.png"),
        )
        plt.close("all")

        run_summary = {
            "label_smoothing": float(label_smoothing),
            "manual_class_weights": dict(best_current_weights),
            "requested_min_severe_threshold": int(cfg_base["min_severe_records"]),
            "effective_min_severe_threshold": int(cv_result["fold_results"][0]["effective_min_severe_threshold"]),
            "n_folds": int(cfg_run["n_splits"]),
            "n_reliable_folds": int(sum(result.get("is_reliable", True) for result in cv_result["fold_results"])),
            "n_record_steps": int(len(raw_oof_table)),
            "best_oof_severe_boost": float(best_boost),
            "raw_oof_metrics": metrics_to_jsonable(raw_metrics),
            "calibrated_oof_metrics": metrics_to_jsonable(calibrated_metrics),
            "fold_summary": cv_result["fold_summary"],
            "per_fold": [metrics_to_jsonable(result) for result in cv_result["fold_results"]],
            "calibration_grid": calibration_search_df.to_dict(orient="records"),
            "output_dir": str(run_output_dir),
        }
        with open(run_output_dir / "results.json", "w") as summary_file:
            json.dump(run_summary, summary_file, indent=2)

        tenfold_tuning_runs.append(
            {
                "class_weight_slug": class_weights_to_slug(best_current_weights),
                "manual_class_weights": dict(best_current_weights),
                "label_smoothing": float(label_smoothing),
                "best_oof_severe_boost": float(best_boost),
                "raw_balanced_accuracy": float(raw_metrics["balanced_accuracy"]),
                "raw_macro_f1": float(raw_metrics["macro_f1"]),
                "calibrated_balanced_accuracy": float(calibrated_metrics["balanced_accuracy"]),
                "calibrated_macro_f1": float(calibrated_metrics["macro_f1"]),
                "calibrated_accuracy": float(calibrated_metrics["accuracy"]),
                "raw_recall_Normal": float(raw_metrics["recall_Normal"]),
                "raw_recall_Mild": float(raw_metrics["recall_Mild"]),
                "raw_recall_Severe": float(raw_metrics["recall_Severe"]),
                "calibrated_recall_Normal": float(calibrated_metrics["recall_Normal"]),
                "calibrated_recall_Mild": float(calibrated_metrics["recall_Mild"]),
                "calibrated_recall_Severe": float(calibrated_metrics["recall_Severe"]),
                "n_reliable_folds": int(run_summary["n_reliable_folds"]),
                "effective_min_severe_threshold": int(run_summary["effective_min_severe_threshold"]),
                "output_dir": str(run_output_dir),
            }
        )
        tenfold_tuning_summaries.append(run_summary)

        print("\nRun complete:")
        print(f"  Label smoothing:          {label_smoothing:.2f}")
        print(f"  Raw OOF balanced acc:     {raw_metrics['balanced_accuracy']:.4f}")
        print(f"  Calibrated OOF bal acc:   {calibrated_metrics['balanced_accuracy']:.4f}")
        reset_tf_state(tf)
        print(f"  Best OOF Severe boost:    {best_boost:.2f}")
        print(f"  Results saved to:         {run_output_dir}")
        gc.collect()
        tf.keras.backend.clear_session()

    summary_df = pd.DataFrame(tenfold_tuning_runs).sort_values(
        [
            "calibrated_balanced_accuracy",
            "calibrated_macro_f1",
            "raw_balanced_accuracy",
            "class_weight_slug",
            "label_smoothing",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    with open(output_root / "label_smoothing_sweep_summary.json", "w") as summary_file:
        json.dump(tenfold_tuning_summaries, summary_file, indent=2)
    summary_df.to_csv(output_root / "label_smoothing_sweep_summary.csv", index=False)

    best_run = summary_df.iloc[0].to_dict()

    print("\n" + "#" * 78)
    print("TENFOLD LABEL-SMOOTHING + OOF CALIBRATION SUMMARY")
    print("#" * 78)
    print(
        summary_df[
            [
                "label_smoothing",
                "best_oof_severe_boost",
                "raw_balanced_accuracy",
                "calibrated_balanced_accuracy",
                "raw_macro_f1",
                "calibrated_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("\nBest setting:")
    print(f"  Label smoothing:        {best_run['label_smoothing']:.2f}")
    print(f"  Best OOF Severe boost:  {best_run['best_oof_severe_boost']:.2f}")
    print(f"  Calibrated bal. acc.:   {best_run['calibrated_balanced_accuracy']:.4f}")
    print(f"  Calibrated macro F1:    {best_run['calibrated_macro_f1']:.4f}")
    print(f"\nSweep summary saved to {output_root}")

    return summary_df, tenfold_tuning_summaries, best_run