import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras import layers
import numpy as np


def transpose_bn_block(inputs, filters, stage, activation="relu"):

    transpose_name = f"decoder_stage_{stage}a_transpose"
    bn_name = f"decoder_stage_{stage}a_bn"
    act_name = f"decoder_stage_{stage}a_{activation}"

    x = inputs
    x = Conv2DTranspose(
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        name=transpose_name,
        use_bias=False,
    )(inputs)

    x = BatchNormalization(axis=-1, name=bn_name)(x)

    x = Activation(activation, name=act_name)(x)

    return x


def conv_bn_block(
    inputs,
    filters,
    stage="",
    activation="relu",
    k_size=3,
    dilation_rate=1,
    use_bias=False,
    name=None,
):

    if name is not None:
        conv_block_name = name

    else:
        conv_block_name = f"decoder_stage_{stage}b"

    x = inputs
    x = Conv2D(
        filters,
        kernel_size=k_size,
        kernel_initializer="he_uniform",
        padding="same",
        use_bias=use_bias,
        dilation_rate=dilation_rate,
        name=f"{conv_block_name}_conv",
    )(x)

    x = BatchNormalization(axis=-1, name=f"{conv_block_name}_bn")(x)

    x = Activation(activation, name=f"{conv_block_name}_{activation}")(x)

    return x


def decoder_block(inputs, filters, stage, skip=None, activation="relu"):

    x = inputs
    x = transpose_bn_block(x, filters, stage, activation)

    if skip is not None:
        x = Concatenate(axis=-1, name=f"decoder_stage_{stage}concat")([x, skip])

    x = conv_bn_block(x, filters, stage, activation)

    return x


def AtrousSpatialPyramidPooling(inputs, dilation_rates=(1, 6, 12, 18)):

    dims = inputs.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    x = conv_bn_block(x, filters=256, k_size=1, use_bias=True, name="ASPP_AvgPool")
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_0 = conv_bn_block(
        inputs, k_size=1, filters=256, dilation_rate=dilation_rates[0], name="ASPP_0"
    )
    out_1 = conv_bn_block(
        inputs, k_size=3, filters=256, dilation_rate=dilation_rates[1], name="ASPP_1"
    )
    out_2 = conv_bn_block(
        inputs, k_size=3, filters=256, dilation_rate=dilation_rates[2], name="ASPP_2"
    )
    out_3 = conv_bn_block(
        inputs, k_size=3, filters=256, dilation_rate=dilation_rates[3], name="ASPP_3"
    )

    x = Concatenate(axis=-1)([out_pool, out_0, out_1, out_2, out_3])
    output = conv_bn_block(x, k_size=1, filters=256, name="ASPP_out")

    return output


class ResidualBlock(layers.Layer):
    def __init__(
        self, output_dim, norm_layer=None, activation="swish", **kwargs
    ) -> None:
        """Standard residual block with pre-activation https://paperswithcode.com/method/residual-block

        Args:
            output_dim (int): The output channel dimension.
            norm_layer (tf.keras.layers.Layer, optional): The normalization layer
                to be applied befor the activation and the convolution. Defaults to None.
                If None, LayerNormalization is applied.
            activation (str, optional): The activation function for the residual.
                Defaults to "swish".
        """
        super().__init__(**kwargs)

        self.outpud_dim = output_dim

        if norm_layer is not None:
            self.norm1 = norm_layer
            self.norm2 = norm_layer
        else:
            self.norm1 = layers.LayerNormalization(epsilon=1e-5)
            self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.act1 = layers.Activation(activation=activation)
        self.conv1 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.act2 = layers.Activation(activation=activation)
        self.conv2 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.proj = layers.Conv2D(output_dim, kernel_size=1)

    def call(self, inputs):

        x = self.norm1(inputs)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.outpud_dim != inputs.shape[-1]:
            inputs = self.proj(inputs)

        return x + inputs


class ConvNextBlock(tf.keras.layers.Layer):
    def __init__(
        self, dim, drop_path=None, layer_scale_init_value=None, name=""
    ) -> None:
        super().__init__(name=name)

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7, padding="same", depth_multiplier=1
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv = tf.keras.layers.Dense(
            units=4 * dim,
        )
        self.act = tf.keras.layers.Activation("gelu")
        self.pw_conv_2 = tf.keras.layers.Dense(
            units=dim,
        )
        if layer_scale_init_value is not None:
            self.gamma = tf.Variable(
                layer_scale_init_value * tf.ones(shape=dim), trainable=True
            )
        else:
            self.gamma = None

        if drop_path is not None:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = None

    def call(self, x):

        input = x

        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv(x)
        x = self.act(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = input + x

        return x


def window_partition(x, window_size):
    """Partitions the image in a number of windows of size window_size

    Args:
        x (tf.tensor): The input image, with shape (B, H, W, C)
        window_size (int): The size of the attention window

    Returns:
        windows: The original tensor reshaped according to the attention windows.
            The new shape is (n_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(
        x, [-1, H // window_size, window_size, W // window_size, window_size, C]
    )

    windows = tf.reshape(
        tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C]
    )

    return windows


def window_reverse(windows, window_size, H, W, C):
    """Reconstructs the original image from the windows partition

    Args:
        windows (tf.tensor): The partitioned windows tensor
        window_size (int): The size of the attention window
        H (int): The height of the image
        W (int): The width of the image

    Returns:
        x (tf.tensor): The input tensor from which the windows partition was obtained.
            The new shape is (B, H, W, C)
    """

    x = tf.reshape(
        windows, [-1, H // window_size, W // window_size, window_size, window_size, C]
    )
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, H, W, C])

    return x


class Mlp(Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        activation="gelu",
        name=None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop = drop

        self.dense1 = Dense(self.hidden_features, name=f"{name}_mlp_dense1")
        self.dense2 = Dense(self.out_features, name=f"{name}_mlp_dense2")
        self.dropout = Dropout(self.drop)
        self.activation = tf.keras.activations.get(activation)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "drop": self.drop,
                "activation_layer": tf.keras.activations.serialize(self.activation),
                "name": self.name,
            }
        )
        return config

    def call(self, x):

        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = self.dim // self.num_heads
        self.qk_scale = qk_scale or head_dim**-0.5

        self.qkv = Dense(
            self.dim * 3, use_bias=self.qkv_bias, name=f"{self.name}_attn_qkv"
        )
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(self.dim, name=f"{self.name}_attn_proj")
        self.proj_drop = Dropout(proj_drop)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "attn_drop": self.attn_drop,
                "proj_drop": self.proj_drop,
                "name": self.name,
            }
        )
        return config

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            f"{self.name}_attn_relative_position_bias_table",
            shape=(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        """
        NOTE: Default indexing for meshgrid in pytorch is ij while in tensorflow ix xy
        so here we have to specify the ij indexing to recover the original implementation
        """
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name=f"{self.name}_attn_relative_position_index",
        )
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(
            tf.reshape(
                self.qkv(x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]
            ),
            perm=[2, 0, 3, 1, 4],
        )  # (3, n_windows*B, n_heads, Wh*Ww, C // n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.qk_scale
        attn = q @ tf.transpose(
            k, perm=[0, 1, 3, 2]
        )  # (n_windows*B, n_heads, Wh*Ww, Wh*Ww)
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, shape=[-1]),
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            shape=[
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ],
        )
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1]
        )  # (n_heads, Wh*Ww, Wh*Ww)
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype
            )
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(Layer):
    def __init__(self, drop_prob, **kwargs) -> None:
        super().__init__(**kwargs)

        self.drop_prob = drop_prob

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})

        return config

    def call(self, x, training=None):

        if self.drop_prob == 0.0 or not training:
            return x

        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(x, keep_prob) * binary_tensor

        return output


class SwinTransformerBlock(Layer):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        activation="gelu",
        name=None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_prob = drop_path_prob

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if self.shift_size < 0 or self.shift_size >= self.window_size:
            raise ValueError(
                "shift size must be greater than zero and smaller than window size"
            )

        self.norm1 = LayerNormalization(epsilon=1e-5, name=f"{self.name}_norm1")
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            name=f"{self.name}_WindowAttention",
        )
        self.drop_path = DropPath(
            self.drop_path_prob if self.drop_path_prob > 0.0 else 0.0
        )
        self.norm2 = LayerNormalization(epsilon=1e-5, name=f"{self.name}_norm2")
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            drop=self.drop,
            name=self.name,
            activation=self.activation,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "input_resolution": self.input_resolution,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path_prob": self.drop_path_prob,
                "activation": self.activation,
                "name": self.name,
            }
        )

        return config

    def build(self, input_shape):

        if self.shift_size > 0:
            # mask for shifted windows
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )

            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name=f"{self.name}_attn_mask",
            )

        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):

        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        if L != H * W:
            raise ValueError("input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C]
        )

        # compute W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C]
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cycle shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(Layer):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(4, 4),
        in_chans=3,
        embed_dim=96,
        name="patch_embed",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = patches_resolution
        self.num_patch = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="proj",
        )

        self.norm_layer = LayerNormalization(epsilon=1e-5, name="norm")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "in_chans": self.in_chans,
                "embed_dim": self.embed_dim,
            }
        )

        return config

    def call(self, x):

        B, H, W, C = x.get_shape().as_list()

        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input tensor shape ({H} x {W}) does not match the model input shape ({self.img_size[0]} x {self.img_size[1]})"
            )

        x = self.proj(x)
        x = tf.reshape(
            x,
            shape=[
                -1,
                (H // self.patch_size[0]) * (W // self.patch_size[1]),
                self.embed_dim,
            ],
        )

        x = self.norm_layer(x)

        return x


class PatchMerging(Layer):
    def __init__(
        self,
        input_resolution,
        dim=None,
        downsample=2,
        name=None,
        **kwargs,
    ) -> None:
        super().__init__(name=f"{name}_PatchMerging_{downsample}X", **kwargs)

        self.input_resolution = input_resolution
        self.downsample = downsample

        if dim is not None:
            self.dim = dim
            self.reduction = Dense(
                self.downsample * dim,
                use_bias=False,
                name=f"{name}_downsample_reduction",
            )
        else:
            self.dim = None
            self.reduction = None

        self.norm = LayerNormalization(epsilon=1e-5, name=f"{name}_downsample_norm")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "dim": self.dim,
                "downsample": self.downsample,
                "name": self.name,
            }
        )
        return config

    def call(self, x):

        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        if L != H * W:
            raise ValueError("input image has wrong size")

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"H ({H}) or W ({W}) are not even")

        x = tf.reshape(x, shape=[-1, H, W, C])

        x_merge = tf.nn.space_to_depth(
            x, block_size=self.downsample, name=f"{self.name}_merge"
        )
        x_merge = tf.reshape(
            x_merge,
            shape=[
                -1,
                (H // self.downsample) * (W // self.downsample),
                C * (self.downsample**2),
            ],
        )

        x_merge = self.norm(x_merge)
        if self.reduction is not None:
            x_merge = self.reduction(x_merge)

        return x_merge


class PatchExpanding(Layer):
    def __init__(
        self,
        input_resolution,
        dim=None,
        upsample=2,
        expand_dims=False,
        name=None,
        **kwargs,
    ) -> None:
        super().__init__(name=f"{name}_PatchExpanding_{upsample}X", **kwargs)

        self.input_resolution = input_resolution
        self.upsample = upsample
        self.expand_dims = expand_dims

        if (dim is not None) and self.expand_dims:
            self.dim = dim
            self.expansion = Dense(
                dim,
                use_bias=False,
                name=f"{name}_upsample_expansion",
            )
        else:
            self.dim = None
            self.expansion = None

        self.norm = LayerNormalization(epsilon=1e-5, name=f"{name}_upsample_norm")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "dim": self.dim,
                "upsample": self.upsample,
                "expand_dims": self.expand_dims,
                "name": self.name,
            }
        )
        return config

    def call(self, x):

        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        if L != H * W:
            raise ValueError("input image has wrong size")

        x = tf.reshape(x, shape=[-1, H, W, C])

        x_merge = tf.nn.depth_to_space(
            x, block_size=self.upsample, name=f"{self.name}_merge"
        )
        x_merge = tf.reshape(
            x_merge,
            shape=[
                -1,
                (H * self.upsample) * (W * self.upsample),
                C // (self.upsample**2),
            ],
        )
        x_merge = self.norm(x_merge)
        if self.expansion is not None:
            x_merge = self.expansion(x_merge)

        return x_merge


def SwinBasicLayer(
    input,
    dim,
    input_resolution,
    depth,
    num_heads,
    window_size,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop=0.0,
    attn_drop=0.0,
    drop_path_prob=0.0,
    downsample=None,
    name=None,
):

    x = input

    for i in range(depth):

        x = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_prob=drop_path_prob[i]
            if isinstance(drop_path_prob, list)
            else drop_path_prob,
            name=f"{name}_block_{i}",
        )(x)

    if downsample is not None:
        x = downsample(input_resolution, dim=dim, name=name)(x)

    return x


def SwinDecoderLayer(
    input,
    skip,
    dim,
    input_resolution,
    depth,
    num_heads,
    window_size,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop=0.0,
    attn_drop=0.0,
    drop_path_prob=0.0,
    upsample=None,
    expand_dims=False,
    name=None,
):

    if upsample is not None:
        x = upsample(
            input_resolution,
            dim=dim,
            expand_dims=expand_dims,
            name=name,
        )(input)
        input_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
    x = Concatenate(axis=-1, name=f"{name}_concat")([x, skip])
    x = Dense(dim, use_bias=False, name=f"{name}_projection")(x)

    for i in range(depth):

        x = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_prob=drop_path_prob[i]
            if isinstance(drop_path_prob, list)
            else drop_path_prob,
            name=f"{name}_block_{i}",
        )(x)

    return x
