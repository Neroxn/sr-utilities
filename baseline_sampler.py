import cv2
import argparse
from PIL import Image
import os

methods = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4,
    'all' : [cv2.INTER_NEAREST,cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for using basic computer vision algorithms to upsample an image"
    )
    parser.add_argument('--img-path', dest="img_path", required=True, help="Path to the image file")
    parser.add_argument('--output-dir', dest="output_dir", required=True, help="Directory to save the upsampled images")
    parser.add_argument('--img-size', dest="img_size", required=True, help="Size to resize the image to either in format 'WIDTHxHEIGHT', e.g., '800x600' or a single multiplier e.g., '4'")
    parser.add_argument('--interpolation', dest="interpolation", choices=['nearest', 'bilinear', 'bicubic', 'lanczos4', 'all'], default='bilinear', help="Interpolation method to use")

    return parser.parse_args()

def resize_image(img, size, interpolation):
    
    results = []
    if isinstance(methods[interpolation],list):
        for ip in methods[interpolation]:
            results.append(cv2.resize(img, size, interpolation=ip))
    else:
        results.append(cv2.resize(img, size, interpolation=methods[interpolation]))
    return results

def main():
    args = parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load image
    image = cv2.imread(args.img_path)

    # Check if img-size contains 'x' or is a single integer
    if 'x' in args.img_size:
        width, height = map(int, args.img_size.split('x'))
    else:
        multiplier = float(args.img_size)
        height, width = image.shape[:2]
        width = int(width * multiplier)
        height = int(height * multiplier)

    resized_images = resize_image(image, (width, height), args.interpolation)

    # Save image with postfix
    for i,resized_image in enumerate(resized_images):
        base_name = os.path.basename(args.img_path)
        root, ext = os.path.splitext(base_name)
        save_path = os.path.join(args.output_dir, f"{root}_{list(methods.keys())[i] if args.interpolation == 'all' else args.interpolation}{ext}")
        cv2.imwrite(save_path, resized_image)
    print(f"Resized image saved at {save_path}")

if __name__ == '__main__':
    main()
