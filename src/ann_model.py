import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_ann(input_dim):

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc")
        ]
    )

    return model


def get_early_stopping():

    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )