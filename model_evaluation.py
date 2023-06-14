import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from segmentation_models.models import Unet
import cv2
import os
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    jaccard_score,
    balanced_accuracy_score,
)
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Evaluate model")

parser.add_argument(
    "--imdir",
    type=str,
    help="Path to the directory containing the image patches (optional), defaults to ./Images",
    default="./Images",
)
parser.add_argument(
    "--maskdir",
    type=str,
    help="Path to the directory containing the mask patches (optional), defaults to ./Masks",
    default="./Masks",
)
parser.add_argument(
    "--savedir",
    type=str,
    help="Path to the directory where the results will be saved (optional), defaults to ./Results",
    default="./Results",
)
parser.add_argument(
    "--save_cm",
    type=bool,
    help="Whether to save the confusion matrix (optional), defaults to True",
    default=True,
)
parser.add_argument(
    "--weights",
    type=str,
    help="Path to the model weights (optional), defaults to ./weights.h5",
    default="./weights.h5",
)
parser.add_argument(
    "--patch_size",
    type=int,
    help="Size of the patches (optional), defaults to 384",
    default=384,
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size (optional), defaults to 8", default=8
)

args = parser.parse_args()


def main(args):

    imdir = args.imdir
    maskdir = args.maskdir
    weights = args.weights
    patch_size = args.patch_size
    batch_size = args.batch_size
    save_cm = args.save_cm
    savedir = args.savedir

    os.makedirs(args.savedir, exist_ok=True)

    # Load the model
    model = Unet(
        (patch_size, patch_size, 3),
        backbone="efficientnetb3",
        classes=7,
        final_activation="softmax",
    )
    model.load_weights(weights)

    # Load the images and masks

    test_imgs = tf.keras.utils.image_dataset_from_directory(
        imdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="rgb",
        seed=42,
    )

    test_masks = tf.keras.utils.image_dataset_from_directory(
        maskdir,
        labels=None,
        batch_size=batch_size,
        image_size=(patch_size, patch_size),
        color_mode="grayscale",
        seed=42,
    )

    test_ds = tf.data.Dataset.zip((test_imgs, test_masks))

    predictions = []
    labels = []

    for x, y in test_ds:

        pred = model.predict(x)

        pred = tf.argmax(pred, axis=-1)

        predictions.append(pred.numpy())

        labels.append(y.numpy())

    prediction_array = np.concatenate(predictions, axis=0)
    prediction_array = np.ravel(prediction_array)

    labels_array = np.concatenate(labels, axis=0)
    labels_array = np.ravel(labels_array)

    f1 = f1_score(y_true=labels_array, y_pred=prediction_array, average="weighted")
    iou = jaccard_score(
        y_true=labels_array, y_pred=prediction_array, average="weighted"
    )
    acc = balanced_accuracy_score(y_true=labels_array, y_pred=prediction_array)

    results = {
        "f1": f1,
        "iou": iou,
        "acc": acc,
    }

    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(os.path.join(savedir, "metrics.csv"), index=False)

    if save_cm:

        names = [
            "Background",
            "WDFP",
            "Swamp",
            "Peat Layer",
            "Fluvial Sands",
            "PDF",
            "Prodelta",
        ]

        cm = confusion_matrix(labels_array, prediction_array, normalize="true")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        plt.rcParams.update({"font.size": 14})

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format=".3f")  # Blues
        ax.set_title("Confusion Matrix", fontsize=24)
        ax.set_ylabel("True Class", fontsize=20)
        ax.set_xlabel("Predicted Class", fontsize=20)

        ax.tick_params(axis="both", labelsize=16)

        plt.savefig(os.path.join(savedir, "confusion_matrix.png"), bbox_inches="tight")


if __name__ == "__main__":

    print("Evaluating model...")

    main(args)

    print("Done!")
