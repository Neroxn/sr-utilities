import os
import random
import shutil
from typing import List

def sample_images(source_dir: str, dest_dir: str, percentage: float) -> List[str]:
    """
    Sample a percentage of image files from source_dir and copy them to dest_dir.

    Args:
        source_dir (str): Source directory containing images.
        dest_dir (str): Destination directory to copy sampled images.
        percentage (float): The percentage of images to sample. Between 0 and 1.

    Returns:
        List[str]: List of copied image file names.
    """
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    sample_count = int(len(image_files) * percentage)
    sampled_files = random.sample(image_files, sample_count)

    for file in sampled_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))

    return sampled_files

if __name__ == "__main__":
    source_dir = "/home/borakargi/dbsr/datasets/dboss_youtube/val_images_hr"
    dest_dir = "/home/borakargi/dbsr/datasets/dboss_youtube/val_images_hr_p002"
    percentage = 0.02  # 20% of the images

    sampled_files = sample_images(source_dir, dest_dir, percentage)
    print(f"Copied {len(sampled_files)} files: {sampled_files}")
