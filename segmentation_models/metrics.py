import tensorflow as tf


EPSILON = tf.keras.backend.epsilon()


class BinaryIoU(tf.keras.metrics.Metric):
    def __init__(self, threshold=None) -> None:
        super(BinaryIoU, self).__init__(name="BinaryIoU")

        self.threshold = threshold
        self.iou = self.add_weight(name="IoU", initializer="zeros")

    def update_state(self, y_true, y_pred):

        score = iou(y_true=y_true, y_pred=y_pred, threshold=self.threshold)

        self.iou.assign_add(score)

    def result(self):

        return self.iou


class MeanIoU(tf.keras.metrics.Metric):
    def __init__(
        self, classes, threshold=None, sparse_y_true=False, sparse_y_pred=False
    ) -> None:
        super(MeanIoU, self).__init__(name="MeanIoU")

        self.threshold = threshold
        self.iou = self.add_weight(name="IoU", initializer="zeros")
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.classes = classes

    def update_state(self, y_true, y_pred):

        if self.sparse_y_pred:
            y_pred = tf.one_hot(y_pred, depth=self.classes)

        if self.sparse_y_true:
            y_true = tf.one_hot(y_true, depth=self.classes)

        score = iou(y_true=y_true, y_pred=y_pred, threshold=self.threshold)

        self.iou.assign_add(score)

    def result(self):

        return self.iou


class FScore(tf.keras.metrics.Metric):
    def __init__(self, threshold=None, beta=1) -> None:
        super(FScore, self).__init__(name="FScore")

        self.threshold = threshold
        self.beta = beta
        self.f_score = self.add_weight(name="f_score", initializer="zeros")

    def update_state(self, y_true, y_pred):

        score = f_score(
            y_true=y_true, y_pred=y_pred, threshold=self.threshold, beta=self.beta
        )

        self.f_score.assign_add(score)

    def result(self):

        return self.f_score


def iou(y_true, y_pred, threshold=None):

    if threshold is not None:

        y_pred = tf.greater(y_pred, threshold)
        y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum((y_true * y_pred), axis=(0, 1, 2))
    union = tf.reduce_sum((y_true + y_pred), axis=(0, 1, 2)) - intersection

    iou = tf.divide((intersection + EPSILON), (union + EPSILON))

    return tf.reduce_mean(iou)


def f_score(y_true, y_pred, threshold=None, beta=1):

    if threshold is not None:

        y_pred = tf.greater(y_pred, threshold)
        y_pred = tf.cast(y_pred, tf.float32)

    true_positive = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
    false_positive = tf.reduce_sum(y_pred, axis=(0, 1, 2)) - true_positive
    false_negative = tf.reduce_sum(y_true, axis=(0, 1, 2)) - true_positive

    f_score = ((1 + beta**2) * true_positive + EPSILON) / (
        (1 + beta**2) * true_positive
        + beta**2 * false_negative
        + false_positive
        + EPSILON
    )
    f_score = tf.reduce_mean(f_score)

    return f_score
