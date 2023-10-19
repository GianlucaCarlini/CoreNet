import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from segmentation_models.utils import predict_big_image
import matplotlib.pyplot as plt
from segmentation_models.models import Unet
from matplotlib.colors import Normalize

palette = np.array(
    [
        [29, 29, 27],  # Background
        [244, 229, 136],  # WDF
        [104, 180, 46],  # Swamp
        [42, 75, 155],  # Organic
        [241, 137, 24],  # Sand
        [128, 192, 123],  # PDF
        [106, 69, 149],  # ProDelta
    ]
)

parser = argparse.ArgumentParser(description="Predict images")

parser.add_argument(
    "--imdir",
    type=str,
    help="Path to the directory containing the images (optional), defaults to ./Images",
    default="./Images",
)
parser.add_argument(
    "--maskdir",
    type=str,
    help="Path to the directory containing the masks (optional), defaults to ./Mask",
    default=None,
)
parser.add_argument(
    "--weights",
    type=str,
    help="Path to the model weights (optional), defaults to ./best_model.h5",
    default="./best_model.h5",
)
parser.add_argument(
    "--saveimdir",
    type=str,
    help="Path to the directory where the predicted images will be saved (optional), defaults to ./Predictions",
    default="./Predictions",
)
parser.add_argument(
    "--target_h",
    type=int,
    help="Target height of the images (optional), defaults to 1538",
    default=1538,
)
parser.add_argument(
    "--target_w",
    type=int,
    help="Target width of the images (optional), defaults to 3074",
    default=3074,
)
parser.add_argument(
    "--patch_size",
    type=int,
    help="Size of the patches (optional), defaults to 384",
    default=384,
)
parser.add_argument(
    "--stride",
    type=int,
    help="Size of the stride (optional), defaults to 96",
    default=96,
)
parser.add_argument(
    "--batch_size", type=int, help="Batch size (optional), defaults to 8", default=8
)
parser.add_argument(
    "--resized_images",
    type=bool,
    help="Whether to save a copy of the resized images (optional), defaults to True",
    default=True,
)
parser.add_argument(
    "--confidence",
    type=bool,
    help="Whether to save the confidence maps (optional), defaults to True",
    default=True,
)
parser.add_argument(
    "--error",
    type=bool,
    help="Whether to save the error maps (optional), defaults to True",
    default=True,
)

args = parser.parse_args()


def quantization(img, palette):
    distance = np.linalg.norm(img[:, :, None] - palette[None, None, :], axis=3)

    quantized = np.argmin(distance, axis=2).astype("uint8")

    return quantized


def main(args):
    imdir = args.imdir
    maskdir = args.maskdir
    weights = args.weights
    saveimdir = args.saveimdir
    patch_size = args.patch_size
    stride = args.stride
    batch_size = args.batch_size
    confidence = args.confidence
    resized_images = args.resized_images
    error = args.error
    target_h = args.target_h
    target_w = args.target_w

    # Load the model
    model = Unet(
        (384, 384, 3), backbone="efficientnetb3", classes=7, final_activation="softmax"
    )
    model.load_weights(weights)

    if error and maskdir is None:
        Warning(
            f"error was set to {error} but maskdir was not provided. No error maps will be saved."
        )

    os.makedirs(os.path.join(saveimdir, "Masks"), exist_ok=True)

    if confidence:
        os.makedirs(os.path.join(saveimdir, "Confidence"), exist_ok=True)

    if error:
        os.makedirs(os.path.join(saveimdir, "Error"), exist_ok=True)

    if resized_images:
        os.makedirs(os.path.join(saveimdir, "Resized_Images"), exist_ok=True)

    # Get the list of images
    imlist = os.listdir(imdir)
    if maskdir is not None:
        masklist = os.listdir(maskdir)
    else:
        masklist = [None for _ in imlist]

    # Loop over the images
    for imname, maskname in zip(imlist, masklist):
        print(f"Predicting {imname} \n")

        # Load the image
        im = cv2.imread(os.path.join(imdir, imname))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h, _, _ = im.shape

        method = "bilinear" if h < target_h else "area"

        im = tf.image.resize_with_pad(
            image=im,
            target_height=target_h,
            target_width=target_w,
            method=method,
            antialias=True,
        )

        im = im[1:-1, 1:-1]
        im = im.numpy()

        if resized_images:
            cv2.imwrite(
                os.path.join(saveimdir, "Resized_Images", imname), im[:, :, ::-1]
            )

        # Predict the image
        pred = predict_big_image(
            img=im,
            model=model,
            patch_size=patch_size,
            stride=stride,
            pad=False,
            classes=7,
            batch_size=batch_size,
        )

        pred_color = palette[np.argmax(pred, axis=-1).astype(np.uint8)]
        pred_color = pred_color[:, :, ::-1]

        # Save the prediction
        cv2.imwrite(os.path.join(saveimdir, "Masks", imname), pred_color)

        # Save the confidence map
        if confidence:
            pred_confidence = np.max(pred, axis=-1)

            norm = Normalize(vmin=pred_confidence.min(), vmax=pred_confidence.max())
            pred_confidence = norm(pred_confidence)

            pred_confidence = plt.cm.afmhot(pred_confidence)[:, :, :3] * 255
            pred_confidence = pred_confidence[:, :, ::-1]

            cv2.imwrite(
                os.path.join(saveimdir, "Confidence", imname),
                pred_confidence.astype(np.uint8),
            )

        # Save the error map
        if error and maskname is not None:
            mask = cv2.imread(os.path.join(maskdir, maskname))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = tf.image.resize_with_pad(
                image=mask,
                target_height=target_h,
                target_width=target_w,
                method="nearest",
            )
            mask = mask[1:-1, 1:-1]
            mask = mask.numpy()
            mask = quantization(mask, palette)
            mask_one_hot = tf.one_hot(mask, depth=7)

            pred_error = tf.keras.metrics.categorical_crossentropy(mask_one_hot, pred)
            pred_error = pred_error.numpy()

            norm = Normalize(vmin=pred_error.min(), vmax=pred_error.max())
            pred_error = norm(pred_error)

            pred_error = plt.cm.afmhot(pred_error)[:, :, :3] * 255
            pred_error = pred_error[:, :, ::-1]

            cv2.imwrite(
                os.path.join(saveimdir, "Error", imname),
                pred_error.astype(np.uint8),
            )


if __name__ == "__main__":
    print("\nPredicting images... \n")

    main(args)

    print("Done!")
