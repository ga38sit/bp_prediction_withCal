import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

@tf.keras.utils.register_keras_serializable()
def tcn_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.0, l2_reg=0.0, name=None):
    shortcut = x

    x = layers.Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv" if name else None
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.Activation("relu", name=f"{name}_relu" if name else None)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop" if name else None)(x)

    # Residual connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, kernel_size=1, padding="same", activation=None,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_proj" if name else None
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn" if name else None)(shortcut)

    x = layers.Add(name=f"{name}_add" if name else None)([x, shortcut])
    return x
    
def build_tcn_withCal_flag(
        input_length=680,
        input_channels=3,
        scalar_features=10,
        n_cal=9,
        cal_label_dim=3, cal_signal_dim=680,
        dropout_rate=0.0,
        l2_reg=0.0,
        useCalData=True,
        *args, **kwargs):

    # --- Input Layers (always present) ---
    signal_in = layers.Input(shape=(input_length, input_channels), name="timeseries")
    scalar_in = layers.Input(shape=(scalar_features,), name="scalar")

    # --- Stem and Residual Blocks (main signal) ---
    x = layers.Conv1D(
        32, kernel_size=15, padding="same", activation=None,
        kernel_regularizer=regularizers.l2(l2_reg), name="stem_conv"
    )(signal_in)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    x = tcn_block(x, 16, dilation_rate=1, dropout_rate=dropout_rate, l2_reg=l2_reg, name="tcn1")
    x = tcn_block(x, 32, dilation_rate=2, dropout_rate=dropout_rate, l2_reg=l2_reg, name="tcn2")
    x = tcn_block(x, 32, dilation_rate=4, dropout_rate=dropout_rate, l2_reg=l2_reg, name="tcn3")
    x = tcn_block(x, 32, dilation_rate=8, dropout_rate=dropout_rate, l2_reg=l2_reg, name="tcn4")

    x = layers.GlobalAveragePooling1D(name="gap_signal")(x)
    x = layers.BatchNormalization(name="signal_bn")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="signal_drop")(x)

    # --- Scalar Feature Processing ---
    s = layers.Dense(
        16, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="scalar_dense"
    )(scalar_in)

    # ---------------------------------------------------------------------
    # Calibration branch (only built/used when useCalData == False)
    # ---------------------------------------------------------------------
    if useCalData and n_cal > 0:
        # --- Calibration Inputs ---
        print("Number of calibration samples (n_cal):", n_cal)
        cal_label_in = layers.Input(shape=(n_cal, cal_label_dim), name="cal_label")
        cal_feat_in = layers.Input(shape=(n_cal, scalar_features), name="cal_feat")
        cal_signal_in = layers.Input(shape=(n_cal, cal_signal_dim), name="cal_signal")  # (B, n_cal, 680)

        # Calibration PPG encoder sub-model: one segment (680,1) -> embedding
        ppg_seg_in = layers.Input(shape=(cal_signal_dim, 1), name="cal_ppg_seg")
        p = layers.Conv1D(
            32, kernel_size=15, padding="same", activation=None,
            kernel_regularizer=regularizers.l2(l2_reg), name="cal_ppg_stem_conv"
        )(ppg_seg_in)
        p = layers.BatchNormalization(name="cal_ppg_stem_bn")(p)
        p = layers.Activation("relu", name="cal_ppg_stem_relu")(p)

        p = tcn_block(p, 16, dilation_rate=1, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_tcn1")
        p = tcn_block(p, 32, dilation_rate=2, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_tcn2")
        p = tcn_block(p, 32, dilation_rate=4, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_tcn3")
        p = tcn_block(p, 32, dilation_rate=8, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_tcn4")

        p = layers.GlobalAveragePooling1D(name="cal_ppg_gap")(p)
        if dropout_rate > 0:
            p = layers.Dropout(dropout_rate, name="cal_ppg_drop")(p)
        cal_ppg_encoder = Model(ppg_seg_in, p, name="cal_ppg_encoder")

        # --- Calibration Signal Processing ---
        cal_signal_4d = layers.Reshape((n_cal, cal_signal_dim, 1), name="cal_signal_reshape")(cal_signal_in)
        cal_ppg_emb = layers.TimeDistributed(cal_ppg_encoder, name="cal_signal_encode")(cal_signal_4d)
        cal_ppg_emb = layers.BatchNormalization(name="cal_ppg_emb_bn")(cal_ppg_emb)

        # --- Calibration Feature Processing (feat + label + ppg embedding) ---
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
            cal_ppg_emb       # (B, n_cal, ppg_emb_dim)
        ])
        cal_h = layers.TimeDistributed(
            layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            name="cal_td_dense"
        )(cal_pair)

        cal_ctx = layers.GlobalAveragePooling1D(name="cal_ctx_pool")(cal_h)
        cal_ctx = layers.BatchNormalization(name="cal_ctx_bn")(cal_ctx)
        if dropout_rate > 0:
            cal_ctx = layers.Dropout(dropout_rate * 0.5, name="cal_ctx_drop")(cal_ctx)

        fused_in = [x, s, cal_ctx]
        inputs = [signal_in, scalar_in, cal_label_in, cal_feat_in, cal_signal_in]
        model_name = "minimal_tcn_with_calibration"

    else:
        fused_in = [x, s]
        inputs = [signal_in, scalar_in]
        model_name = "minimal_tcn_no_calibration"


    # --- Fusion Layers ---
    fused = layers.Concatenate(name="fusion_concat")(fused_in)
    fused = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fusion_dense1"
    )(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    if dropout_rate > 0:
        fused = layers.Dropout(dropout_rate, name="fusion_drop")(fused)

    # --- Output Layers ---

    label_dbp = layers.Dense(1, name="label_dbp")(fused)
    label_sbp = layers.Dense(1, name="label_sbp")(fused)
    label_delta = layers.Dense(1, name="label_delta")(fused)


    model = Model(
        inputs=inputs,
        outputs={
            "label_dbp": label_dbp,
            "label_sbp": label_sbp,
            "label_delta": label_delta,
        },
        name=model_name
    )
    return model