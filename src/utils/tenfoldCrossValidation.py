import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K

class tenfoldCrossValidation:
    def __init__(self, model_fn, sampler, trainer, evaluator,
                 n_iterations=10, n_splits=10):
        self.model_fn = model_fn
        self.sampler = sampler
        self.trainer = trainer
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        self.n_splits = n_splits

    def _print_split_info(self, y_train, y_test):
        """
        Correct one-hot mapping based on LabelEncoder:
        y[:, 0] = abnormal indicator (1 if abnormal, 0 if normal)
        y[:, 1] = normal indicator (1 if normal, 0 if abnormal)
        """
        n_abnormal_train = (y_train[:, 0] == 1).sum()
        n_normal_train = (y_train[:, 1] == 1).sum()
        n_abnormal_test = (y_test[:, 0] == 1).sum()
        n_normal_test = (y_test[:, 1] == 1).sum()

        print(f"Train: Normal={n_normal_train}, Abnormal={n_abnormal_train}, Total={len(y_train)}")
        print(f"Test:  Normal={n_normal_test}, Abnormal={n_abnormal_test}, Total={len(y_test)}")

        return n_normal_train, n_abnormal_train

    def _evaluate_fold(self, model, X_train, y_train, X_test, y_test, verbose=True):
        n_normal_train, n_abnormal_train = self._print_split_info(y_train, y_test)

        # Match notebook class weights exactly
        class_weights = {
            0: 1.0,
            1: n_normal_train / n_abnormal_train
        }
        if verbose:
            print(f"Class weights: 0={class_weights[0]:.2f}, 1={class_weights[1]:.2f}")

        # Train
        if verbose:
            print("Training...", end=" ", flush=True)

        history = self.trainer.train(model, X_train, y_train, class_weights)

        if verbose:
            n_epochs = len(history.history["loss"])
            final_loss = history.history["loss"][-1]
            print(f"Done! ({n_epochs} epochs, final loss={final_loss:.4f})")

        # Predict - evaluate ABNORMAL detection (class 0) to match notebook intent
        y_prob_raw = model.predict(X_test, verbose=0)
        y_prob = y_prob_raw[:, 0]  # Probability of class 0 (abnormal)

        # Ground truth labels - abnormal indicator
        y_test_int = y_test[:, 0]  # Abnormal indicator (0 or 1)

        # Evaluate
        fold_results = self.evaluator.evaluate_with_print(y_test_int, y_prob, verbose=verbose)

        return fold_results

    def run(self, X, y_onehot, y_int, normalize_fn):
        """
        Correct one-hot mapping:
        - y_onehot[:, 0] = abnormal indicator
        - y_onehot[:, 1] = normal indicator
        - y_int: 0 = abnormal, 1 = normal
        """
        all_results = []

        for iteration in range(1, self.n_iterations + 1):
            print(f"\nIteration {iteration}/{self.n_iterations}")

            # Balance dataset (undersample normal to match abnormal)
            X_balanced, y_balanced_int = self.sampler.balance(
                X, y_int, normalize_fn, seed=42 + iteration
            )

            # Convert to one-hot (matches notebook's y_balanced)
            num_classes = y_onehot.shape[1] if y_onehot is not None else 2
            y_balanced = np.eye(num_classes)[y_balanced_int]

            # y_labels_balanced for stratification (normal indicator)
            y_labels_balanced = y_balanced[:, 1]

            # 10-fold CV
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=42 + iteration
            )

            for fold, (train_idx, test_idx) in enumerate(skf.split(X_balanced, y_labels_balanced), 1):
                print(f"\n--- Fold {fold}/{self.n_splits} ---")

                X_train = X_balanced[train_idx]
                y_train = y_balanced[train_idx]
                X_test = X_balanced[test_idx]
                y_test = y_balanced[test_idx]

                # Build model and clear session (matches notebook)
                K.clear_session()
                model = self.model_fn()

                # Evaluate fold
                fold_results = self._evaluate_fold(model, X_train, y_train, X_test, y_test)
                all_results.append(fold_results)

        return all_results