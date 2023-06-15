import tensorflow as tf
import numpy as np
import pandas as pd
from segmentation_models.models import Unet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from datetime import datetime
import os

today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trainimdir",
    type=str,
    help="Path to directory where the train image patches are saved (optional), default is ./train_image_patches",
    default="./train_image_patches",
)
parser.add_argument(
    "--trainmaskdir",
    type=str,
    help="Path to directory where the train mask patches are saved (optional), default is ./train_masks_patches",
    default="./train_masks_patches",
)
parser.add_argument(
    "--valimdir",
    type=str,
    help="Path to directory where the validation image patches are saved (optional), default is ./val_images_patches",
    default="./val_images_patches",
)
parser.add_argument(
    "--valmaskdir",
    type=str,
    help="Path to directory where the validation mask patches are saved (optional), default is ./val_masks_patches",
    default="./val_masks_patches",
)
parser.add_argument(
    "--backbone",
    type=str,
    help="Backbone to use for the model (optional), default is efficientnetb3",
    default="efficientnetb3",
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size (optional), defaults to 8", default=8
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs (optional), defaults to 100",
    default=100,
)
parser.add_argument(
    "--lr", type=float, help="Learning rate (optional), defaults to 1e-4", default=1e-4
)
parser.add_argument(
    "--final_lr",
    type=float,
    help="Final learning rate (optional), defaults to 5e-6",
    default=5e-6,
)
parser.add_argument(
    "--patch_size",
    type=int,
    help="Size of the patches (optional), defaults to 384",
    default=384,
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to save the model checkpoints (optional), defaults to ./ckpt",
    default="./ckpt",
)
parser.add_argument(
    "--history_path",
    type=str,
    help="Path to save the training history (optional), defaults to ./history",
    default="./history",
)

args = parser.parse_args()


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()

        self.rotate_inputs = tf.keras.layers.RandomRotation(
            factor=1.0, fill_mode="constant", seed=seed
        )
        self.rotate_labels = tf.keras.layers.RandomRotation(
            factor=1.0, fill_mode="constant", seed=seed
        )

        self.rnd_contrast = tf.keras.layers.RandomContrast(factor=0.05, seed=seed)
        self.rnd_bright = tf.keras.layers.RandomBrightness(factor=0.05, seed=seed)

    def call(self, inputs, labels):

        inputs = self.rotate_inputs(inputs)
        labels = self.rotate_labels(labels)
        inputs = self.rnd_contrast(inputs)
        inputs = self.rnd_bright(inputs)

        return inputs, labels


def main(args):

    train_imdir = args.trainimdir
    train_maskdir = args.trainmaskdir
    val_imdir = args.valimdir
    val_maskdir = args.valmaskdir
    backbone = args.backbone
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    final_lr = args.final_lr
    patch_size = args.patch_size
    ckpt_path = args.ckpt_path
    history_path = args.history_path

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(history_path, exist_ok=True)

    # Load the train and validation datasets
    train_imgs = tf.keras.utils.image_dataset_from_directory(
        train_imdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="rgb",
        seed=42,
    )
    train_masks = tf.keras.utils.image_dataset_from_directory(
        train_maskdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="grayscale",
        seed=42,
    )
    val_imgs = tf.keras.utils.image_dataset_from_directory(
        val_imdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="rgb",
        seed=42,
    )
    val_masks = tf.keras.utils.image_dataset_from_directory(
        val_maskdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="grayscale",
        seed=42,
    )

    train_ds = tf.data.Dataset.zip((train_imgs, train_masks))
    val_ds = tf.data.Dataset.zip((val_imgs, val_masks))

    train_ds = train_ds.map(Augment())

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    decay_steps = train_ds.cardinality().numpy() * epochs
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        lr, decay_steps, final_lr, power=0.5
    )

    # Define the model
    model = Unet(
        (384, 384, 3), backbone="efficientnetb3", classes=7, final_activation="softmax"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate_fn,
            name="Adam",
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[MyMeanIOU(num_classes=7)],
    )

    model_checkpoint = ModelCheckpoint(
        os.path.join(ckpt_path, f"{backbone}_{today}.h5"),
        monitor="val_my_mean_iou",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint],
    )

    # Save the training history
    pd.DataFrame(history.history).to_csv(
        os.path.join(history_path, f"{backbone}_{today}.csv"), index=False
    )


if __name__ == "__main__":

    print("Training the model... \n")

    print(f"EPOCHS: {args.epochs} \n")
    print(f"LR: {args.lr} \n")
    print(f"FINAL_LR: {args.final_lr} \n")
    print(f"PATCH_SIZE: {args.patch_size} \n")
    print(f"BATCH_SIZE: {args.batch_size} \n")

    main(args)

    print("Done!")
