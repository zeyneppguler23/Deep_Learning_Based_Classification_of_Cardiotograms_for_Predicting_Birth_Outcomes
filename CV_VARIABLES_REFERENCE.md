# Cross-Validation Loop - Variables Reference

## Outer Loop Variables (Iteration)

| Variable       | Type | Meaning                                                                     |
| -------------- | ---- | --------------------------------------------------------------------------- |
| `iteration`    | int  | Current iteration number (1 to N_ITERATIONS=5). Labels independent CV runs. |
| `N_ITERATIONS` | int  | Total number of independent CV experiments (default: 5)                     |

## K-Fold Split Variables

| Variable    | Type            | Meaning                                                                                 |
| ----------- | --------------- | --------------------------------------------------------------------------------------- |
| `fold`      | int             | Current fold number within an iteration (1 to N_FOLDS=5).                               |
| `N_FOLDS`   | int             | Number of folds per iteration (default: 5)                                              |
| `train_idx` | ndarray         | Indices of samples assigned to training set in this fold. Shape (n_train,)              |
| `test_idx`  | ndarray         | Indices of samples assigned to test set in this fold. Shape (n_test,)                   |
| `skf`       | StratifiedKFold | Sklearn object that generates stratified train/test splits                              |
| `y_strat`   | ndarray         | Stratification variable: maps -1→3 so all classes are represented in splits. Shape (N,) |

## Input Data Variables (Original - All Data)

| Variable     | Type    | Meaning                                                                                                                           |
| ------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `X`          | ndarray | Input signal data. Shape (N, 2, 1800, 1): N samples, 2 channels (FHR+UC), 1800 timesteps, 1 feature. Range: normalized (z-score). |
| `y_ord`      | ndarray | Ordinal targets for severity. Shape (N, 2). Values in {0, 1}. [p(>=Mild), p(>=Severe)]                                            |
| `y_interp_f` | ndarray | Interpretability targets (float). Shape (N, 1). Values in {0, 1}. 0=uninterpretable, 1=interpretable                              |
| `y_sev`      | ndarray | Severity labels (original). Shape (N,). Values in {-1, 0, 1, 2}. -1=uninterpretable, 0=Normal, 1=Mild, 2=Severe                   |
| `y_interp`   | ndarray | Interpretability labels (int). Shape (N,). Values in {0, 1}. Derived from y_sev.                                                  |

## Training Set Variables (Per Fold)

| Variable         | Type    | Meaning                                                                                                |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `X_train`        | ndarray | Training signals for this fold. Shape (n_train, 2, 1800, 1). Subset of X.                              |
| `y_ord_train`    | ndarray | Ordinal targets for training set. Shape (n_train, 2). Subset of y_ord.                                 |
| `y_interp_train` | ndarray | Interpretability targets for training. Shape (n_train, 1). Subset of y_interp_f.                       |
| `y_sev_train`    | ndarray | Severity labels for training. Shape (n_train,). Subset of y_sev. Original with -1 for uninterpretable. |

## Test Set Variables (Per Fold)

| Variable        | Type    | Meaning                                                                                           |
| --------------- | ------- | ------------------------------------------------------------------------------------------------- |
| `X_test`        | ndarray | Test signals for this fold. Shape (n_test, 2, 1800, 1). Subset of X.                              |
| `y_ord_test`    | ndarray | Ordinal targets for test set. Shape (n_test, 2). Subset of y_ord.                                 |
| `y_interp_test` | ndarray | Interpretability targets for test. Shape (n_test, 1). Subset of y_interp_f.                       |
| `y_sev_test`    | ndarray | Severity labels for test. Shape (n_test,). Subset of y_sev. Original with -1 for uninterpretable. |

## Sample Weight Variables (Class Balancing)

| Variable         | Type    | Meaning                                                                                                                                                                  |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `w_interp_train` | ndarray | Sample weights for interpretability task. Shape (n_train,). All ones (no class weighting currently).                                                                     |
| `w_sev_base`     | ndarray | Base weights for severity: 1.0 if interpretable, 0.0 if uninterpretable. Shape (n_train,). Masks out uninterpretable samples.                                            |
| `w_sev_train`    | ndarray | Final weights for severity task. Shape (n_train,). Values: 1.0 (Normal), 0.8 (Mild), 6.0 (Severe), 0.0 (uninterpretable). Applied per-sample to balance class imbalance. |
| `class_counts`   | ndarray | Count of each severity class in training set. Shape (3,). [count_Normal, count_Mild, count_Severe]. Only interpretable samples.                                          |
| `total_interp`   | int     | Total number of interpretable samples in training fold. Used for inverse frequency weighting (not currently used).                                                       |

## Class Distribution Variables (Per Fold)

Training set:
| Variable | Type | Meaning |
|----------|------|---------|
| `n_normal_train` | int | Count of Normal severity samples in training set. |
| `n_mild_train` | int | Count of Mild severity samples in training set. |
| `n_severe_train` | int | Count of Severe severity samples in training set. |
| `n_unint_train` | int | Count of uninterpretable samples in training set. |

Test set:
| Variable | Type | Meaning |
|----------|------|---------|
| `n_normal_test` | int | Count of Normal severity samples in test set. |
| `n_mild_test` | int | Count of Mild severity samples in test set. |
| `n_severe_test` | int | Count of Severe severity samples in test set. |
| `n_unint_test` | int | Count of uninterpretable samples in test set. |

## Model & Training Variables

| Variable        | Type          | Meaning                                                                                               |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| `model`         | Keras Model   | The compiled multi-task neural network. Two outputs: [severity_ord, interpretable]. Rebuilt per fold. |
| `LEARNING_RATE` | float         | Adam optimizer learning rate. Default: 1e-4.                                                          |
| `batch_size`    | int           | Training batch size. Default: 8.                                                                      |
| `epochs`        | int           | Maximum training epochs. Default: 300. Early stopping may end earlier.                                |
| `history`       | Keras History | Training history object. Contains loss curves and other metrics per epoch.                            |

## Callbacks Variables

| Variable     | Type              | Meaning                                                                                          |
| ------------ | ----------------- | ------------------------------------------------------------------------------------------------ |
| `early_stop` | EarlyStopping     | Callback that stops training if loss doesn't improve for 30 epochs. Restores best weights.       |
| `reduce_lr`  | ReduceLROnPlateau | Callback that reduces learning rate by 0.5x if loss doesn't improve for 15 epochs. Min LR: 1e-7. |

## Training History Storage

| Variable                  | Type | Meaning                                                       |
| ------------------------- | ---- | ------------------------------------------------------------- |
| `history.history['loss']` | list | Loss values per epoch for this fold. Length = epochs trained. |

## Threshold Tuning Variables (Training Set)

| Variable                    | Type    | Meaning                                                                                                               |
| --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `sev_pred_train_all`        | ndarray | Severity ordinal predictions on TRAINING set. Shape (n_train, 2). Values in [0,1] (sigmoid outputs).                  |
| `mask_train_interp`         | ndarray | Boolean mask: True if sample is interpretable. Shape (n_train,). Used to filter training data.                        |
| `sev_pred_train_interp`     | ndarray | Severity predictions for interpretable training samples only. Shape (n_train_interp, 2). Filtered by mask.            |
| `y_sev_train_interp_labels` | ndarray | Severity labels for interpretable training samples. Shape (n_train_interp,). Values in {0, 1, 2}.                     |
| `best_t1`                   | float   | Optimal threshold for P(class >= 1). Found via grid search. Range: 0.30-0.50. Tuned on training set.                  |
| `best_t2`                   | float   | Optimal threshold for P(class >= 2). Found via grid search. Range: 0.20-0.40. Tuned on training set.                  |
| `best_qwk_train`            | float   | Quadratic Weighted Kappa score on training set with tuned thresholds. Used to select best thresholds. Range: [-1, 1]. |

## Test Set Predictions

| Variable       | Type    | Meaning                                                                                                |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `sev_pred_ord` | ndarray | Severity ordinal predictions on TEST set. Shape (n_test, 2). Values in [0,1] (sigmoid outputs).        |
| `interp_pred`  | ndarray | Interpretability predictions on TEST set. Shape (n_test, 1). Values in [0,1] (sigmoid output).         |
| `interp_prob`  | ndarray | Interpretability probabilities (reshaped). Shape (n_test,). Values in [0,1].                           |
| `interp_hat`   | ndarray | Interpretability class predictions (binary). Shape (n_test,). Values in {0, 1}. 0.5 threshold applied. |

## Threshold Application Variables (Test Set)

| Variable              | Type    | Meaning                                                                                              |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `mask_test_interp`    | ndarray | Boolean mask: True if test sample is interpretable. Shape (n_test,).                                 |
| `y_sev_test_interp`   | ndarray | Severity labels for interpretable test samples only. Shape (n_test_interp,). Values in {0, 1, 2}.    |
| `sev_pred_ord_interp` | ndarray | Severity predictions for interpretable test samples. Shape (n_test_interp, 2). Filtered by mask.     |
| `y_pred_sev_tuned`    | ndarray | Predicted severity class labels using tuned thresholds. Shape (n_test_interp,). Values in {0, 1, 2}. |

## Evaluation Metrics - Interpretability Task

| Variable     | Type  | Meaning                                                                                            |
| ------------ | ----- | -------------------------------------------------------------------------------------------------- |
| `interp_acc` | float | Accuracy on interpretability task for this fold. Range: [0, 1].                                    |
| `interp_f1`  | float | F1-score on interpretability task for this fold. Range: [0, 1]. Macro average (only 2 classes).    |
| `interp_auc` | float | AUC-ROC on interpretability task for this fold. Range: [0, 1]. Set to NaN if only 1 class present. |

## Evaluation Metrics - Severity Task

| Variable  | Type  | Meaning                                                                                                        |
| --------- | ----- | -------------------------------------------------------------------------------------------------------------- |
| `sev_acc` | float | Accuracy on severity task for interpretable samples only. Range: [0, 1].                                       |
| `sev_f1`  | float | Macro F1-score on severity task. Range: [0, 1]. Macro average across 3 classes.                                |
| `sev_qwk` | float | Quadratic Weighted Kappa on severity task. Range: [-1, 1]. Ordinal agreement measure. 1.0 = perfect agreement. |

## Results Storage Container

| Variable     | Type | Meaning                                                                                              |
| ------------ | ---- | ---------------------------------------------------------------------------------------------------- |
| `cv_results` | dict | Dictionary storing all cross-validation results. Persists across all folds and iterations. Contains: |
|              |      | - `interp`: Interpretability metrics (accuracy, F1, AUC, predictions)                                |
|              |      | - `severity`: Severity metrics (accuracy, F1-macro, QWK, predictions)                                |
|              |      | - `training`: Training history (loss curves, epochs, fold/iteration labels)                          |

---

## Constants/Hyperparameters

| Variable        | Type  | Value      | Meaning                                                             |
| --------------- | ----- | ---------- | ------------------------------------------------------------------- |
| `LEARNING_RATE` | float | 1e-4       | Adam optimizer learning rate                                        |
| `NORMAL_WEIGHT` | float | 1.0        | Manual weight for Normal severity class                             |
| `MILD_WEIGHT`   | float | 0.8        | Manual weight for Mild severity class                               |
| `SEVERE_WEIGHT` | float | 6.0        | Manual weight for Severe severity class (high to balance imbalance) |
| `N_ITERATIONS`  | int   | 5          | Number of independent CV runs                                       |
| `N_FOLDS`       | int   | 5          | Number of folds per iteration                                       |
| Loss weights    | list  | [1.0, 0.5] | [severity_loss_weight, interpretability_loss_weight]                |

---

## Key Relationships

```
Data Flow:
X, y_sev, y_ord, y_interp_f, y_interp
    ↓
For each iteration:
    ↓
StratifiedKFold split (5 folds)
    ↓
For each fold:
    X_train, y_*_train ←→ X_test, y_*_test
    (with weights: w_sev_train, w_interp_train)
    ↓
Train model
    ↓
Predict on train set → tune thresholds (best_t1, best_t2)
    ↓
Predict on test set → apply thresholds → compute metrics
    ↓
Store in cv_results[fold]
    ↓
Aggregate cv_results across all 25 folds (5 iterations × 5 folds)
```

## Shape Summary

| Variable       | Shape           | Note                                    |
| -------------- | --------------- | --------------------------------------- |
| X              | (N, 2, 1800, 1) | N samples, 2 channels, 1800 timesteps   |
| y_sev          | (N,)            | -1, 0, 1, 2                             |
| y_interp       | (N,)            | 0 or 1                                  |
| y_ord          | (N, 2)          | Ordinal targets                         |
| y_interp_f     | (N, 1)          | Float version of y_interp               |
| sev_pred_ord   | (n_test, 2)     | Model output: [P(≥Mild), P(≥Severe)]    |
| interp_pred    | (n_test, 1)     | Model output: [P(interpretable)]        |
| w_sev_train    | (n_train,)      | Per-sample weights for severity         |
| w_interp_train | (n_train,)      | Per-sample weights for interpretability |
