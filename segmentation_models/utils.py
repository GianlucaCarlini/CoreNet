import tensorflow as tf
import numpy as np
import time
from typing import Callable


def predict_big_image(img, patch_size, stride, classes, model, batch_size=8, pad=False):
    """Function to predict gigantic images. The purpose of this function is to predict
       very big images without using an excessive amount of RAM.

    Args:
        img (uint8): The image to predict.
        patch_size (int): The size of the image patches to feed the model.
        stride (int): The stride of the sliding window producing the patches.
        classes (int): The number of classes predicted by the model.
        model (tf.Model): The tensorflow model.
        batch_size (int, optional): The number of patches to fit in a batch. Defaults to 8.
        pad (bool, optional): Wether to pad or not the images with zeros. Defaults to False.
            If False, the image is not padded, so only the valid portions of the image are used.
            If True, the image is padded with zeros and the entire image is predicted.
            Note: For memory efficiency the code works on the image inplace, so if pad is False
            the original image will be cut to the valid values. If pad is True, instead,
            the original dimensions of the image are mantained.

    Returns:
        model_prediction(np.array): The prediction of the model
    """
    h, w, c = img.shape

    if pad:
        if img.shape[0] % patch_size == 0:
            pad_width = 0
        else:
            pad_width = int(
                (patch_size * ((img.shape[0] // patch_size) + 1) - img.shape[0])
            )
        if img.shape[1] % patch_size == 0:
            pad_height = 0
        else:
            pad_height = int(
                (patch_size * ((img.shape[1] // patch_size) + 1) - img.shape[1])
            )
        img = np.pad(img, [(0, pad_width), (0, pad_height), (0, 0)])
    else:
        img = img[
            : patch_size * (img.shape[0] // patch_size),
            : patch_size * (img.shape[1] // patch_size),
        ]

    model_prediction = np.zeros(
        shape=(img.shape[0], img.shape[1], classes), dtype=np.float16
    )
    mask = np.zeros(shape=(img.shape[0], img.shape[1], classes), dtype=np.uint8)

    N_PATCH_W = 1 + (img.shape[1] - patch_size) // stride
    N_PATCH_H = 1 + (img.shape[0] - patch_size) // stride
    BATCH_SIZE = batch_size

    batch = np.zeros(shape=(BATCH_SIZE, patch_size, patch_size, c))
    n = 0

    """
    ----------------------
    PREDICTION LOOP
    ----------------------
    """

    start = time.time()
    for i in range(N_PATCH_H):
        for j in range(0, BATCH_SIZE * (N_PATCH_W // BATCH_SIZE), BATCH_SIZE):
            """
            Here we create the batch to feed the network
            """
            for k in range(BATCH_SIZE):
                batch[k, ...] = img[
                    i * stride : i * stride + patch_size,
                    (j + k) * stride : (j + k) * stride + patch_size,
                ]

            batch = batch.astype(np.float32)
            prediction = model.predict_on_batch(batch)

            """
            Here we assign the predictions
            """
            for k in range(BATCH_SIZE):
                model_prediction[
                    i * stride : i * stride + patch_size,
                    (j + k) * stride : (j + k) * stride + patch_size,
                ] += prediction[k, ...]
                mask[
                    i * stride : i * stride + patch_size,
                    (j + k) * stride : (j + k) * stride + patch_size,
                ] += 1
                n += 1
                if n % 100 == 0:
                    print(f"predicted {n} patches out of {N_PATCH_H * N_PATCH_W}")

        """
        Now we have to predict the last batch which is not a multiple of BATCH_SIZE
        """
        for j in range(N_PATCH_W % BATCH_SIZE):
            batch[j, ...] = img[
                i * stride : i * stride + patch_size,
                (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride : (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride
                + patch_size,
            ]
        batch = batch.astype(np.float32)
        prediction = model.predict_on_batch(batch[: (N_PATCH_W % BATCH_SIZE), ...])

        for j in range(N_PATCH_W % BATCH_SIZE):
            model_prediction[
                i * stride : i * stride + patch_size,
                (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride : (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride
                + patch_size,
            ] += prediction[j, ...]
            mask[
                i * stride : i * stride + patch_size,
                (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride : (BATCH_SIZE * (N_PATCH_W // BATCH_SIZE) + j)
                * stride
                + patch_size,
            ] += 1
            n += 1
            if n % 100 == 0:
                print(f"predicted {n} patches out of {N_PATCH_H * N_PATCH_W}")
    end = time.time()
    print(f"prediction lasted {end - start}")

    if pad:
        img = img[:h, :w]

    model_prediction *= 1.0 / mask

    return model_prediction


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        decay_schedule_fn: Callable,
        start_lr: float,
        target_lr: float,
        warmup_steps: int,
        name: str = None,
    ):
        """Wrapper function to add a warmup start to a tensorflow decay schedule

        Args:
            decay_schedule_fn (Callable): Tensorflow LearningRateSchedule function or
                a custom Schedule with a __call__ method that accepts a step
            start_lr (float): Initial learning rate
            target_lr (float): Learning rate to reach at the end of the warmup
            warmup_steps (int): Number of warmup steps
            name (str, optional): Name of the object. Defaults to None.
        """
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_lr = (self.target_lr - self.start_lr) * (
                global_step_float / warmup_steps_float
            ) + self.start_lr

            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_lr,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "start_lr": self.start_lr,
            "decay_schedule_fn": self.decay_schedule_fn,
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }
