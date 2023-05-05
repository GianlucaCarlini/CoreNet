import tensorflow as tf
import functools


class ModelBackbones:

    _backbone_layers = {
        "efficientnetb0": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb1": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb2": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb3": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb4": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb5": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb6": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "efficientnetb7": (
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ),
        "resnet50": (
            "conv4_block6_out",
            "conv3_block4_out",
            "conv2_block3_out",
            "conv1_relu",
        ),
        "resnet101": (
            "conv4_block23_out",
            "conv3_block4_out",
            "conv2_block3_out",
            "conv1_relu",
        ),
        "mobilenetv1": (
            "conv_pw_11_relu",
            "conv_pw_5_relu",
            "conv_pw_3_relu",
            "conv_pw_1_relu",
        ),
        "mobilenetv2": (
            "block_13_expand_relu",
            "block_6_expand_relu",
            "block_3_expand_relu",
            "block_1_expand_relu",
        ),
    }

    _models = {
        "efficientnetb0": tf.keras.applications.efficientnet.EfficientNetB0,
        "efficientnetb1": tf.keras.applications.efficientnet.EfficientNetB1,
        "efficientnetb2": tf.keras.applications.efficientnet.EfficientNetB2,
        "efficientnetb3": tf.keras.applications.efficientnet.EfficientNetB3,
        "efficientnetb4": tf.keras.applications.efficientnet.EfficientNetB4,
        "efficientnetb5": tf.keras.applications.efficientnet.EfficientNetB5,
        "efficientnetb6": tf.keras.applications.efficientnet.EfficientNetB6,
        "efficientnetb7": tf.keras.applications.efficientnet.EfficientNetB7,
        "resnet50": tf.keras.applications.resnet.ResNet50,
        "resnet101": tf.keras.applications.resnet.ResNet101,
        "mobilenetv1": tf.keras.applications.mobilenet.MobileNet,
        "mobilenetv2": tf.keras.applications.mobilenet_v2.MobileNetV2,
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    @staticmethod
    def get_kwargs():
        return {}

    def inject_submodules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            modules_kwargs = self.get_kwargs()
            new_kwargs = dict(list(kwargs.items()) + list(modules_kwargs.items()))
            return func(*args, **new_kwargs)

        return wrapper

    def get(self, name):

        if name not in self.models_names():
            raise ValueError(
                "No such model `{}`, available models: {}".format(
                    name, list(self.models_names())
                )
            )

        model_fn = self.models[name]
        model_fn = self.inject_submodules(model_fn)
        return model_fn

    def get_feature_layers(self, name):
        return self._backbone_layers[name]

    def get_backbone(self, name, *args, **kwargs):

        model_fn = self.get(name)
        model = model_fn(*args, **kwargs)
        return model


Backbones = ModelBackbones()
