from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Trainer:
    def __init__(self, batch_size=8, max_epochs=300):
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # Fixed callbacks matching notebook exactly
        self.callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=25,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=0
            )
        ]

    def train(self, model, X_train, y_train, class_weights):
        """
        Train a compiled model using fixed settings.
        Assumes model is already compiled.
        Matches notebook exactly.
        """
        history = model.fit(
            X_train, y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=self.callbacks,
            verbose=0
        )
        return history