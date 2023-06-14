import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="Save patches from images and masks")

parser.add_argument(
    "--imdir",
    type=str,
    help="Path to the directory containing the images (optional), defaults to ./Images",
    default="./Images",
)
parser.add_argument(
    "--maskdir",
    type=str,
    help="Path to the directory containing the masks (optional), defaults to ./Masks",
    default="./Masks",
)
parser.add_argument(
    "--saveimdir",
    type=str,
    help="Path to the directory where the image patches will be saved (optional), defaults to ./Images_patches",
    default="./Images_patches",
)
parser.add_argument(
    "--savemaskdir",
    type=str,
    help="Path to the directory where the mask patches will be saved (optional), defaults to ./Masks_patches",
    default="./Masks_patches",
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
    help="Size of the stride (optional), defaults to 384",
    default=384,
)
parser.add_argument(
    "--thresh",
    type=float,
    help="Threshold to exclude patches that are mostly background (optional), defaults to 0.1",
    default=0.1,
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

args = parser.parse_args()


def quantization(img, palette):

    distance = np.linalg.norm(img[:, :, None] - palette[None, None, :], axis=3)

    quantized = np.argmin(distance, axis=2).astype("uint8")

    return quantized


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


def save_patches(img, patch_size, mask, savedirs, stride=None, thresh=0.1):
    """Generates pathces of a given size from an image and saves them to disk

    Args:
        img (np.array): The image to be patched
        patch_size (int): The size of the patch
        mask (np.array): The segmentation mask associated with the image.
        savedirs (list): A list of two paths were image and mask patches are saved.
        stride (int, optional): The size of the stride to generate patches. Defaults to None.
        thresh (float, optional): The threshold to exclude patches that are mostly background.
            The patch is saved if the number of foreground pixels is higher then thres in percentage.
            Defaults to 0.1.
    """

    if stride is None:
        stride = patch_size

    h, w, _ = img.shape

    n_patch_h = ((h - patch_size) // stride) + 1
    n_patch_w = ((w - patch_size) // stride) + 1

    n_patch = n_patch_w * n_patch_h

    y = np.arange(n_patch_h)
    x = np.arange(n_patch_w)
    xy = np.meshgrid(x, y)

    x = xy[0].ravel() * stride
    y = xy[1].ravel() * stride
    x2 = x + patch_size
    y2 = y + patch_size

    for i in range(n_patch):

        patch_img = img[y[i] : y2[i], x[i] : x2[i]].astype(np.uint8)
        patch_mask = mask[y[i] : y2[i], x[i] : x2[i]].astype(np.uint8)

        if np.count_nonzero(patch_mask) > thresh * patch_size**2:
            cv2.imwrite(f"{savedirs[0]}_{i}.png", patch_img)
            cv2.imwrite(f"{savedirs[1]}_{i}.png", patch_mask)


def main(args):

    imdir = args.imdir
    maskdir = args.maskdir
    saveimdir = args.saveimdir
    savemaskdir = args.savemaskdir
    patch_size = args.patch_size
    stride = args.stride
    thresh = args.thresh
    target_h = args.target_h
    target_w = args.target_w

    os.makedirs(saveimdir, exist_ok=True)
    os.makedirs(savemaskdir, exist_ok=True)

    imgs = os.listdir(imdir)
    masks = os.listdir(maskdir)

    for im, msk in zip(imgs, masks):

        img = cv2.imread(os.path.join(imdir, im))
        mask = cv2.imread(os.path.join(maskdir, msk))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        method = "bilinear" if h < target_h else "area"

        img = tf.image.resize_with_pad(
            img,
            target_height=target_h,
            target_width=target_w,
            method=method,
            antialias=True,
        )
        mask = tf.image.resize_with_pad(
            mask, target_height=target_h, target_width=target_w, method="nearest"
        )

        img = img.numpy().squeeze()
        mask = mask.numpy().squeeze()

        img = img[1:-1, 1:-1]
        mask = mask[1:-1, 1:-1]

        quant = quantization(mask, palette)

        save_patches(
            img,
            patch_size=patch_size,
            mask=quant,
            stride=stride,
            thresh=thresh,
            savedirs=[
                os.path.join(saveimdir, im[:-4]),
                os.path.join(savemaskdir, msk[:-4]),
            ],
        )


if __name__ == "__main__":

    print("creating patches... \n")
    print(f"patch size: {args.patch_size}, stride: {args.stride} \n")

    main(args)

    print("done! \n")
