from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Trainer:
    def __init__(self, batch_size=8, max_epochs=300):
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # Fixed callbacks for all models
        self.callbacks = [
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=15, min_lr=1e-7, verbose=1),
            EarlyStopping(monitor="loss", patience=25, restore_best_weights=True, verbose=1)
        ]

    def train(self, model, X_train, y_train, class_weights):
        """
        Train a compiled model using fixed settings.
        Assumes model is already compiled.
        """
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            class_weight=class_weights,
            callbacks=self.callbacks,
            verbose=1
        )
        return history
