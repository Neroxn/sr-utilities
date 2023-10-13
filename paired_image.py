import os
import cv2
import lmdb
import argparse
import numpy as np
from typing import List, Tuple

def make_lmdb(directory: str, output_path: str) -> None:
    """
    Convert the images in the directory to LMDB format.

    Args:
        directory (str): Directory containing the images.
        output_path (str): Path to save the LMDB database.

    Returns:
        None
    """

    file_list = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]

    # Determine the size of the database
    map_size = sum([os.path.getsize(os.path.join(directory, f)) for f in file_list]) * 10

    env = lmdb.open(output_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, img_file in enumerate(file_list):
            img_path = os.path.join(directory, img_file)
            with open(img_path, "rb") as f:
                img_data = f.read()
            txn.put(f"{idx:08}".encode("utf-8"), img_data)

def process_images(input_folder: str, scale: float, use_lmdb: bool, output_folder: str) -> None:
    """
    Downsample images and save them, with the option to use LMDB.

    Args:
        input_folder (str): Path to the input images.
        scale (float): Scale by which to downsample.
        use_lmdb (bool): Whether to save images as LMDB.
        output_folder (str): Path to save processed images and metadata.

    Returns:
        None
    """
    if not output_folder:
        print("Warning: No output folder provided. Saving results in the input folder.")
        output_folder = input_folder

    gt_folder = os.path.join(output_folder, "gt")
    lr_folder = os.path.join(output_folder, "lr")

    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    gt_metadata = []
    lr_metadata = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):
            filepath = os.path.join(input_folder, filename)

            # Read the image
            image = cv2.imread(filepath)
            
            # Create GT metadata
            h, w, c = image.shape
            gt_metadata.append(f"{filename} ({h},{w},{c}) 1")

            # Save original image to GT folder
            gt_path = os.path.join(gt_folder, filename)
            cv2.imwrite(gt_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            # Resize the image
            lr_image = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Create LR metadata
            h, w, c = lr_image.shape
            lr_metadata.append(f"{filename} ({h},{w},{c}) 1")

            # Save the downscaled image to LR folder
            lr_path = os.path.join(lr_folder, filename)
            cv2.imwrite(lr_path, lr_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # Write metadata to files
    with open(os.path.join(gt_folder, "metadata.txt"), "w") as file:
        file.write("\n".join(gt_metadata))

    with open(os.path.join(lr_folder, "metadata.txt"), "w") as file:
        file.write("\n".join(lr_metadata))

    if use_lmdb:
        make_lmdb(gt_folder, os.path.join(output_folder, "gt.lmdb"))
        make_lmdb(lr_folder, os.path.join(output_folder, "lr.lmdb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing utility")
    parser.add_argument('input', type=str, help='Path to the input images')
    parser.add_argument('scale', type=float, help='Downsampling scale')
    parser.add_argument('--lmdb', action='store_true', help='Use LMDB format for saving')
    parser.add_argument('--output', type=str, default=None, help='Output folder to save results')

    args = parser.parse_args()

    process_images(args.input, args.scale, args.lmdb, args.output)