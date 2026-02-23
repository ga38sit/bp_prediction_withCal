import tensorflow as tf
from typing import List, Optional


def build_mlp_regressor_og(
    scalar_features: int,
    hidden_dims: List[int] = [2048, 4096, 8192, 2048],
    activation: str = "relu",
    dropout: float = 0,
    layer_norm: bool = False,

    *args, **kwargs
):

    input_shape = (scalar_features,)
    print(f"Building MLP model with input shape: {input_shape}")

    scalar_in = tf.keras.layers.Input(shape=(scalar_features,), name="scalar")

    x = scalar_in
    model_inputs = [scalar_in]

    # --- main MLP trunk ---
    for i, h in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(h, activation=activation, name=f"dense_{i}")(x)
        if layer_norm:
            x = tf.keras.layers.LayerNormalization(name=f"ln_{i}")(x)
        if dropout and dropout > 0.0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i}")(x)

    label_delta = tf.keras.layers.Dense(1, name="label_delta")(x)
    label_dbp = tf.keras.layers.Dense(1, name="label_dbp")(x)
    label_sbp = tf.keras.layers.Dense(1, name="label_sbp")(x)

    model = tf.keras.Model(
        inputs=model_inputs,
        outputs={"label_delta": label_delta, "label_dbp": label_dbp, "label_sbp": label_sbp},
        name="MLPRegressorOG",
    )
    return model