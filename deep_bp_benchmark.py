
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, stride: int = 1, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.stride = stride

        # Main path
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=stride, padding="same", name=f"{self.name}_conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn1")
        self.relu1 = tf.keras.layers.ReLU(name=f"{self.name}_relu1")

        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=1, padding="same", name=f"{self.name}_conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn2")

        # Projection created in build when needed
        self.proj = None
        self.proj_bn = None

        self.add = tf.keras.layers.Add(name=f"{self.name}_add")
        self.out_relu = tf.keras.layers.ReLU(name=f"{self.name}_out")

    def build(self, input_shape):
        in_filters = int(input_shape[-1])
        if in_filters != self.filters or self.stride != 1:
            self.proj = tf.keras.layers.Conv1D(self.filters, kernel_size=1, strides=self.stride, padding="same", name=f"{self.name}_proj")
            self.proj_bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_proj_bn")
        super().build(input_shape)

    def call(self, x, training=None):
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y, training=training)

        shortcut = x
        if self.proj is not None:
            shortcut = self.proj(shortcut)
            shortcut = self.proj_bn(shortcut, training=training)

        y = self.add([shortcut, y])
        y = self.out_relu(y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "stride": self.stride})
        return config


@tf.keras.utils.register_keras_serializable()
class BPHead(tf.keras.layers.Layer):
    def __init__(self, name_prefix: str, name: str = None, **kwargs):

        super().__init__(name=name or name_prefix, **kwargs)
        self.name_prefix = name_prefix

        self.conv64 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", name=f"{self.name_prefix}_conv64")
        self.conv2 = tf.keras.layers.Conv1D(2, kernel_size=3, padding="same", name=f"{self.name_prefix}_conv2")
        self.gap = tf.keras.layers.GlobalAveragePooling1D(name=f"{self.name_prefix}_gap")
        self.fc = tf.keras.layers.Dense(1, name=f"{self.name_prefix}_fc")

    def call(self, x, training=None):
        h = self.conv64(x)
        h = self.conv2(h)
        h = self.gap(h)
        h = self.fc(h)
        return h

    def get_config(self):
        config = super().get_config()
        config.update({"name_prefix": self.name_prefix})
        return config

     
def build_deep_bp_og(
        input_length=400,
        input_channels=1,

        *args, **kwargs
    ):


    input_shape = (input_length, input_channels)
    print(f"Building Deep-BP OG model with input shape: {input_shape}")

    layers = tf.keras.layers

    # -----------------------
    # Main input + backbone
    # -----------------------
    signal_in = layers.Input(shape=(input_length, input_channels), name="timeseries")
    x = signal_in

    x = layers.Conv1D(filters=2, kernel_size=3, padding="same", name="stem_conv1")(x)
    x = layers.BatchNormalization(name="stem_bn1")(x)
    x = layers.ReLU(name="stem_relu1")(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", name="stem_conv2")(x)
    x = layers.BatchNormalization(name="stem_bn2")(x)
    x = layers.ReLU(name="stem_relu2")(x)

    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool")(x)

    x = ResidualBlock(filters=64, stride=1, name="res1")(x)
    x = ResidualBlock(filters=128, stride=2, name="res2")(x)
    x = ResidualBlock(filters=256, stride=2, name="res3")(x)
    x = ResidualBlock(filters=512, stride=2, name="res4")(x)

    inputs = [signal_in]
    model_name = "DeepBP"

    label_dbp = BPHead(name_prefix="label_dbp")(x)
    label_sbp = BPHead(name_prefix="label_sbp")(x)
    label_delta = BPHead(name_prefix="label_delta")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "label_delta": label_delta,
            "label_dbp": label_dbp,
            "label_sbp": label_sbp,
        },
        name=model_name
    )
    return model