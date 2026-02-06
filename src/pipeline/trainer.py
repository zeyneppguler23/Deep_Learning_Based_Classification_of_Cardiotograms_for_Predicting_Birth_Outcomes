from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K

class CTGTrainer:
    def __init__(self, model_builder, lr=1e-3, batch_size=8, max_epochs=300):
        self.model_builder = model_builder
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def train(self, X_train, y_train):
        K.clear_session()
        model = self.model_builder()

        model.compile(
            optimizer=Adam(self.lr),
            loss="categorical_crossentropy"
        )

        early_stop = EarlyStopping(
            monitor="loss", patience=25, restore_best_weights=True, verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=15, min_lr=1e-7, verbose=0
        )

        history = model.fit(
            X_train, y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        return model, history
