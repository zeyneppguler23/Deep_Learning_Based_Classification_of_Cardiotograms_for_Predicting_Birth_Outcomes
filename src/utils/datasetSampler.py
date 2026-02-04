import numpy as np
from sklearn.utils import resample

class DatasetSampler:
    def __init__(self, normal_label=1, abnormal_label=0):
        self.normal_label = normal_label
        self.abnormal_label = abnormal_label

    def balance(self, X, y_int, normalize_fn, seed):
        """
        Return (X_bal, y_bal_int).
        Downsample/upsample the majority class to match minority size.
        Uses replacement only when majority size < minority size.
        """
        normal_idx = np.where(y_int == self.normal_label)[0]
        abnormal_idx = np.where(y_int == self.abnormal_label)[0]

        # If expected labels missing, fallback to generic two-class handling
        if len(normal_idx) == 0 or len(abnormal_idx) == 0:
            classes, counts = np.unique(y_int, return_counts=True)
            if len(classes) < 2:
                raise ValueError("Need at least two classes to balance.")
            class_indices = {cls: np.where(y_int == cls)[0] for cls in classes}
            minority_cls = classes[np.argmin(counts)]
            majority_cls = classes[np.argmax(counts)]
            minority_idx = class_indices[minority_cls]
            majority_idx = class_indices[majority_cls]
        else:
            # identify minority / majority
            if len(normal_idx) <= len(abnormal_idx):
                minority_idx, majority_idx = normal_idx, abnormal_idx
            else:
                minority_idx, majority_idx = abnormal_idx, normal_idx

        n_min = len(minority_idx)
        if n_min == 0:
            raise ValueError("No samples found for minority class; cannot balance.")

        replace = len(majority_idx) < n_min
        sampled_majority = resample(
            majority_idx,
            n_samples=n_min,
            replace=replace,
            random_state=seed
        )

        balanced_idx = np.concatenate([sampled_majority, minority_idx])
        # use numpy Generator (default_rng) to avoid RandomState linter warning
        rng = np.random.default_rng(seed)
        balanced_idx = rng.permutation(balanced_idx)

        X_bal = normalize_fn(X[balanced_idx])
        y_bal = y_int[balanced_idx]

        return X_bal, y_bal
