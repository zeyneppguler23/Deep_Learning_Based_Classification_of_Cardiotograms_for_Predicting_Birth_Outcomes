from tensorflow.keras.optimizers import Adam

def compileModel(model):
    """
    Compile model with fixed optimizer, learning rate, loss, and metrics.
    Used for all models for consistency.
    """
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
