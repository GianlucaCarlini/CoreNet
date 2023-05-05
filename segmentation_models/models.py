from .blocks import decoder_block
from .blocks import conv_bn_block
from .blocks import AtrousSpatialPyramidPooling
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from .backbones import Backbones


def Unet(
    input_shape,
    backbone="efficientnetb3",
    classes=1,
    decoder_activation="relu",
    final_activation="sigmoid",
    filters=[256, 128, 64, 32, 16],
):
    """Instantiates a Unet-like segmentation model with custom backbones.

    Args:
        input_shape (tuple): The shape of the input tensor in the format HxWxC
        backbone (str, optional): The backbone of the model. Defaults to "efficientnetb3".
        classes (int, optional): Number of classes to predict. Determines the output
            channel dimension of the model. Defaults to 1.
        decoder_activation (str, optional): The activation function of the decoder
            convolutional blocks. Defaults to "relu".
        final_activation (str, optional): Activation function of the output layer.
            Defaults to "sigmoid".
        filters (list, optional): Number of filters of the successive decoder blocks,
            starting from the deeper blocks (HxW / 16) up to the shallower (HxW).
            Defaults to [256, 128, 64, 32, 16].

    Returns:
        tf.keras.Model: The segmentation model.
    """

    encoder = Backbones.get_backbone(
        backbone, include_top=False, input_shape=input_shape
    )
    layers = Backbones.get_feature_layers(backbone)

    x = encoder.output

    skip_connections = []
    for layer in layers:
        skip_connections.append(encoder.get_layer(layer).output)

    for i, skip in enumerate(skip_connections):
        if backbone == "efficientnetv2_b3" and i > 1:
            skip = BatchNormalization(axis=-1, name=f"BatchNorm_{i}")(skip)
            skip = Activation("swish", name=f"Activation_{i}")(skip)
        x = decoder_block(
            inputs=x,
            filters=filters[i],
            stage=i,
            skip=skip,
            activation=decoder_activation,
        )

    x = decoder_block(
        inputs=x, filters=filters[-1], stage=4, activation=decoder_activation
    )

    x = Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding="same",
        name="final_conv",
        dtype="float32",
    )(x)
    x = Activation(final_activation, name=final_activation, dtype="float32")(x)

    model = tf.keras.models.Model(encoder.input, x)

    return model


def DeepLabV3Plus(
    input_shape,
    classes=1,
    final_activation="sigmoid",
    dilation_rates=(1, 6, 12, 18),
    backbone="resnet101",
):
    """Instantiates a DeepLabV3+ model with custom backbones

    Args:
        input_shape (tuple): Shape of the input tensor in format HxWxC
        classes (int, optional): Number of classes to predict. Determines the output
            channel dimension of the model. Defaults to 1.
        final_activation (str, optional): Activation function of the output layer.
            Defaults to "sigmoid".
        dilation_rates (tuple, optional): Dilation rates of the ASPP layer.
            Defaults to (1, 6, 12, 18).
        backbone (str, optional): The backbone of the model. Defaults to "resnet101".

    Returns:
        tf.keras.Model: The segmentation model
    """

    model_input = tf.keras.Input(shape=input_shape)

    encoder = Backbones.get_backbone(
        backbone, input_tensor=model_input, include_top=False
    )
    layers = Backbones.get_feature_layers(backbone)

    x = encoder.get_layer(layers[0]).output
    x = AtrousSpatialPyramidPooling(x, dilation_rates=dilation_rates)

    input_a = UpSampling2D(
        size=(input_shape[0] // 4 // x.shape[1], input_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = encoder.get_layer(layers[2]).output
    input_b = conv_bn_block(input_b, filters=48, k_size=1, name="decoder_stage0")

    x = Concatenate(axis=-1)([input_a, input_b])
    x = conv_bn_block(x, filters=256, k_size=3, name="decoder_stage1_0")
    x = conv_bn_block(x, filters=256, k_size=3, name="decoder_stage1_1")
    x = UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = Conv2D(classes, kernel_size=1, padding="same", dtype="float32")(x)

    x = Activation(final_activation, name=final_activation, dtype="float32")(x)

    return tf.keras.Model(inputs=model_input, outputs=x)
