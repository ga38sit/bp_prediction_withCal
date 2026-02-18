import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def build_mlp_withCal(
        scalar_features=15,
        n_cal=9,
        cal_label_dim=3,
        dropout_rate=0.01,
        l2_reg=0.0,
        useCalData=True,   # True => use calibration inputs/branch; False => no calibration
        *args, **kwargs):

    # --- Input Layers (always present) ---
    scalar_in = layers.Input(shape=(scalar_features,), name="scalar")

    # --- Scalar Feature Processing ---
    s = layers.Dense(
        16, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="scalar_dense1"
    )(scalar_in)
    s = layers.BatchNormalization(name="scalar_bn1")(s)
    if dropout_rate > 0:
        s = layers.Dropout(dropout_rate, name="scalar_drop1")(s)

    # ---------------------------------------------------------------------
    # Calibration branch (only when useCalData == True)
    # ---------------------------------------------------------------------
    if useCalData and n_cal > 0:
        print("Number of calibration samples (n_cal):", n_cal)
        cal_label_in = layers.Input(shape=(n_cal, cal_label_dim), name="cal_label")
        cal_feat_in = layers.Input(shape=(n_cal, scalar_features), name="cal_feat")

        # --- Calibration Feature Processing ---
        print(cal_label_in.shape)
        cal_dbp = cal_label_in[..., 0:1]  # (B, n_cal, 1)
        cal_sbp = cal_label_in[..., 1:2]  # (B, n_cal, 1)
        cal_pp = cal_label_in[..., 2:3]   # (B, n_cal, 1)
        
        print(cal_dbp.shape)
        cal_pair = layers.Concatenate(axis=-1, name="cal_pair")([
            cal_feat_in,      # (B, n_cal, scalar_features)
            cal_dbp,          # (B, n_cal, 1) - DBP labels
            cal_sbp,          # (B, n_cal, 1) - SBP labels
            cal_pp,           # (B, n_cal, 1) - Pulse pressure (derived feature)
        ])
        cal_h = layers.TimeDistributed(
            layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            name="cal_td_dense"
        )(cal_pair)

        cal_ctx = layers.GlobalAveragePooling1D(name="cal_ctx_pool")(cal_h)
        cal_ctx = layers.BatchNormalization(name="cal_ctx_bn")(cal_ctx)
        if dropout_rate > 0:
            cal_ctx = layers.Dropout(dropout_rate * 0.5, name="cal_ctx_drop")(cal_ctx)

        fusion_inputs = [s, cal_ctx]
        model_inputs = [scalar_in, cal_label_in, cal_feat_in]
        model_name = "mlp_with_calibration"
    else:
        fusion_inputs = [s]
        model_inputs = [scalar_in]
        model_name = "mlp_no_calibration"

    # --- Fusion Layers ---
    fused = layers.Concatenate(name="fusion_concat")(fusion_inputs) if len(fusion_inputs) > 1 else fusion_inputs[0]
    fused = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fusion_dense"
    )(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    if dropout_rate > 0:
        fused = layers.Dropout(dropout_rate, name="fusion_drop")(fused)

    # --- Output Layers ---
    label_dbp = layers.Dense(1, name="label_dbp")(fused)
    label_sbp = layers.Dense(1, name="label_sbp")(fused)
    label_delta = layers.Dense(1, name="label_delta")(fused)

    model = Model(
        inputs=model_inputs,
        outputs={
            "label_dbp": label_dbp,
            "label_sbp": label_sbp,
            "label_delta": label_delta,
        },
        name=model_name
    )
    return model