import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

   
@tf.keras.utils.register_keras_serializable()
def res_block_1d(x, filters, kernel_size=5, dropout_rate=0.0, l2_reg=0.0, name=None):
    shortcut = x

    x = layers.Conv1D(
        filters, kernel_size, padding="same", activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv1" if name else None
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = layers.Activation("relu", name=f"{name}_relu1" if name else None)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop1" if name else None)(x)


    x = layers.Conv1D(
        filters, kernel_size, padding="same", activation=None,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_conv2" if name else None
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, kernel_size=1, padding="same", activation=None,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_proj" if name else None
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn" if name else None)(shortcut)

    x = layers.Add(name=f"{name}_add" if name else None)([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu_out" if name else None)(x)
    return x

def build_minimal_resnet_withCal_and_signal_bn_v2withCalFlag(
        input_length=680,
        input_channels=3,
        scalar_features=15,
        n_cal=9,                # change here between 3 and 6 and 9
        cal_label_dim=3, cal_signal_dim=680,
        dropout_rate=0.05,
        l2_reg=0.0,
        useCalData=True,    # True => use calibration inputs/branch; False => no calibration
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

    x = res_block_1d(x, 16, kernel_size=9, dropout_rate=dropout_rate, l2_reg=l2_reg, name="res1")
    x = res_block_1d(x, 32, kernel_size=7, dropout_rate=dropout_rate, l2_reg=l2_reg, name="res2")

    x = layers.GlobalAveragePooling1D(name="gap_signal")(x)
    x = layers.BatchNormalization(name="signal_bn")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="signal_drop")(x)

    # --- Scalar Feature Processing ---
    s = layers.Dense(
        16, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="scalar_dense1"
    )(scalar_in)

    # ---------------------------------------------------------------------
    # Calibration branch (only when useCalData == True)
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

        p = layers.MaxPooling1D(2, name="cal_ppg_stem_pool")(p)
        p = res_block_1d(p, filters=16, kernel_size=9, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_resblock1")
        p = res_block_1d(p, filters=32, kernel_size=7, dropout_rate=dropout_rate, l2_reg=l2_reg, name="cal_ppg_resblock2")

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

        fusion_inputs = [x, s, cal_ctx]
        model_inputs = [signal_in, scalar_in, cal_label_in, cal_feat_in, cal_signal_in]
        model_name = "minimal_resnet_with_calibration_bn_v2"
    else:
        fusion_inputs = [x, s]
        model_inputs = [signal_in, scalar_in]
        model_name = "minimal_resnet_no_calibration_bn_v2"

    # --- Fusion Layers ---
    fused = layers.Concatenate(name="fusion_concat")(fusion_inputs)
    fused = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fusion_dense"
    )(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    if dropout_rate > 0:
        fused = layers.Dropout(dropout_rate, name="fusion_drop")(fused)

    # --- Output Layers ---
    # label_mean = layers.Dense(2, name="label_mean")(fused)
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