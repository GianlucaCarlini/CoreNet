import tensorflow as tf
from .metrics import iou, f_score


class JaccardLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):

        return 1 - iou(y_true=y_true, y_pred=y_pred, threshold=None)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, beta=1, name="DiceLoss") -> None:
        super().__init__(name=name)

        self.beta = beta

    def call(self, y_true, y_pred):

        return 1 - f_score(y_true=y_true, y_pred=y_pred, threshold=None, beta=self.beta)
