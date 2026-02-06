from sklearn.model_selection import StratifiedKFold

class CrossValidator:
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def split(self, X, y_labels, seed):
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=seed
        )
        return skf.split(X, y_labels)
