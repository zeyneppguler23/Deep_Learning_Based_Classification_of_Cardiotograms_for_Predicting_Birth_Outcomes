import numpy as np
from keras.utils import to_categorical
from pipeline.trainer import CTGTrainer
from pipeline.CrossValidator import CrossValidator
from pipeline.evaluator import BinaryClassifierEvaluator

class CTGExperiment:
    def __init__(
        self,
        model_builder,
        sampler,
        evaluator,
        normalizer_fn,
        n_iterations=10,
        n_folds=10,
        base_seed=42
    ):
        self.sampler = sampler
        self.trainer = CTGTrainer(model_builder)
        self.evaluator = evaluator
        self.cv = CrossValidator(n_folds)
        self.normalizer_fn = normalizer_fn
        self.n_iterations = n_iterations
        self.base_seed = base_seed

    def run(self, X, y_onehot, y_int):
        self.evaluator.reset()

        for iteration in range(1, self.n_iterations + 1):
            print(f"\nIteration {iteration}/{self.n_iterations}")

            #  Step 1: Undersample
            X_bal, y_bal_int = self.sampler.sample(
                X, y_int, seed=self.base_seed + iteration
            )

            #  Step 2: Normalize
            X_bal = self.normalizer_fn(X_bal)

            #  Step 3: One-hot
            y_bal = to_categorical(y_bal_int, num_classes=2)
            y_labels_balanced = y_bal[:, 1]

            #  Step 4: Stratified 10-fold CV
            for fold, (train_idx, test_idx) in enumerate(
                self.cv.split(X_bal, y_labels_balanced, seed=self.base_seed + iteration),
                1
            ):
                print(f"\n--- Fold {fold}/10 ---")

                X_train = X_bal[train_idx]
                y_train = y_bal[train_idx]
                X_test = X_bal[test_idx]
                y_test = y_bal[test_idx]

                #  Print split info (exactly like your notebook)
                n_normal_train = (y_train[:, 0] == 1).sum()
                n_abnormal_train = (y_train[:, 1] == 1).sum()
                n_normal_test = (y_test[:, 0] == 1).sum()
                n_abnormal_test = (y_test[:, 1] == 1).sum()

                print(f"Train: Normal={n_normal_train}, Abnormal={n_abnormal_train}, Total={len(train_idx)}")
                print(f"Test:  Normal={n_normal_test}, Abnormal={n_abnormal_test}, Total={len(test_idx)}")

                #  Train
                print("Training...", end=" ", flush=True)
                model, history = self.trainer.train(X_train, y_train)
                print(f"Done! ({len(history.history['loss'])} epochs, final loss={history.history['loss'][-1]:.4f})")

                #  Predict
                y_prob = model.predict(X_test, verbose=0)[:, 1]
                y_test_int = y_test[:, 1]

                print(
                    f"Predictions: min={y_prob.min():.3f}, "
                    f"max={y_prob.max():.3f}, "
                    f"mean={y_prob.mean():.3f}, "
                    f"std={y_prob.std():.3f}"
                )

                #  Evaluate
                fold_results = self.evaluator.evaluate_fold(
                    y_test_int, y_prob, iteration=iteration, fold=fold
                )

                print(f"AUC: {fold_results['auc']:.3f}")
                for thr in self.evaluator.thresholds:
                    print(
                        f"Thr: {thr:.3f} | "
                        f"F1: {fold_results[f'f1@{thr}']:.3f} | "
                        f"Prec: {fold_results[f'precision@{thr}']:.3f} | "
                        f"Rec: {fold_results[f'recall@{thr}']:.3f} | "
                        f"Sens: {fold_results[f'sensitivity@{thr}']:.3f} | "
                        f"Acc: {fold_results[f'accuracy@{thr}']:.3f}"
                    )

        return self.evaluator.fold_results
