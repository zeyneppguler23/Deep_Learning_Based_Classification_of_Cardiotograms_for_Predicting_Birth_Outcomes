from sklearn.utils import resample
import numpy as np

class BalancedSampler:
    def sample(self, X, y_int, seed):
        abnormal_indices = np.where(y_int == 0)[0]
        normal_indices = np.where(y_int == 1)[0]

        sampled_normal_indices = resample(
            normal_indices,
            n_samples=len(abnormal_indices),
            replace=False,
            random_state=seed
        )

        sampled_abnormal_indices = abnormal_indices.copy()
        balanced_indices = np.concatenate([sampled_normal_indices, sampled_abnormal_indices])

        return X[balanced_indices], y_int[balanced_indices]
