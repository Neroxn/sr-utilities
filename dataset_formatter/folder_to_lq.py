import cv2
import os
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def list_image_files(folder_path: str) -> List[str]:
    """List image files in a given folder."""
    extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

def downscale_image(image, scale: float, interpolation_type: int) -> None:
    """Downscale a given image."""
    h, w = image.shape[:2]
    new_dim = (int(w / scale), int(h / scale))
    return cv2.resize(image, new_dim, interpolation=interpolation_type)

def process_image(img_path: str, output_folder: str, scale: float, postfix: str, prefix: str, interpolation_mapping: Dict[str, int], interpolation_type: str) -> None:
    img = cv2.imread(img_path)
    img_downscaled = downscale_image(img, scale, interpolation_mapping[interpolation_type])
    
    filename = os.path.basename(img_path)
    filename_without_ext = os.path.splitext(filename)[0]
    filename_extension = filename.split('.')[-1]
    new_filename = f"{prefix}{filename_without_ext}{postfix}.{filename_extension}"
    output_path = os.path.join(output_folder, new_filename)
    
    cv2.imwrite(output_path, img_downscaled)

def folder_to_lq(input_folder: str, output_folder: str, scale: float, postfix: str, prefix: str, interpolation_type: str, n_threads: int) -> None:
    """Downscale all images in a folder."""
    interpolation_mapping: Dict[str, int] = {
        'linear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos4': cv2.INTER_LANCZOS4
    }
    
    if interpolation_type not in interpolation_mapping:
        print(f"Invalid interpolation type. Available options: {', '.join(interpolation_mapping.keys())}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = list_image_files(input_folder)
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(tqdm(executor.map(lambda x: process_image(x, output_folder, scale, postfix, prefix, interpolation_mapping, interpolation_type), image_files), total=len(image_files)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downscale images in a folder.')
    parser.add_argument('input_folder', type=str, help='Input folder path')
    parser.add_argument('output_folder', type=str, help='Output folder path')
    parser.add_argument('--scale', type=float, default=4.0, help='Downscaling factor')
    parser.add_argument('--postfix', type=str, default='', help='Postfix to add to filenames')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to add to filenames')
    parser.add_argument('--interpolation_type', type=str, default='linear', help='Interpolation type')
    parser.add_argument('--n_threads', type=int, default=4, help='Number of threads to use')
    
    args = parser.parse_args()
    
    folder_to_lq(args.input_folder, args.output_folder, args.scale, args.postfix, args.prefix, args.interpolation_type, args.n_threads)
