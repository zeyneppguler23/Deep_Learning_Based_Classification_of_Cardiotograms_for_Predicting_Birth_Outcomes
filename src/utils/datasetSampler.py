import numpy as np
from sklearn.utils import resample

class DatasetSampler:
    def __init__(self, normal_label=1, abnormal_label=0):
        """
        Integer label encoding:
        - 0 = abnormal
        - 1 = normal
        
        After one-hot encoding with np.eye(2)[y_int]:
        - y[:, 0] = abnormal indicator (1 if abnormal, 0 otherwise)
        - y[:, 1] = normal indicator (1 if normal, 0 otherwise)
        """
        self.normal_label = normal_label
        self.abnormal_label = abnormal_label

    def balance(self, X, y_int, normalize_fn, seed):
        """
        Return (X_bal, y_bal_int).
        Undersample normal cases (majority) to match abnormal count (minority).
        Uses ALL abnormal cases.
        
        Dataset has: 164 normal (label=1), 56 abnormal (label=0)
        Result: 56 normal + 56 abnormal = 112 balanced samples
        """
        normal_idx = np.where(y_int == self.normal_label)[0]
        abnormal_idx = np.where(y_int == self.abnormal_label)[0]

        if len(normal_idx) == 0 or len(abnormal_idx) == 0:
            raise ValueError("Need both normal and abnormal samples to balance.")

        # Undersample normal cases to match abnormal count (matches notebook logic)
        np.random.seed(seed)
        sampled_normal_indices = resample(
            normal_idx,
            n_samples=len(abnormal_idx),  # Match abnormal count (56)
            replace=False,
            random_state=seed
        )
        
        # Use ALL abnormal cases
        sampled_abnormal_indices = abnormal_idx.copy()
        
        # Combine into balanced dataset
        balanced_indices = np.concatenate([sampled_normal_indices, sampled_abnormal_indices])
        X_balanced = normalize_fn(X[balanced_indices])
        
        # Return integer labels
        y_balanced_int = y_int[balanced_indices]
        
        return X_balanced, y_balanced_int