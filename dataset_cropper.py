#############################################################
# Author @ Bora KARGI                                       #
# A script for randomly cropping COCO-formatted datasets    #
#   to create pairs of SR images                            #
#############################################################
import argparse
import os
import cv2
import numpy as np
from pycocotools.coco import COCO

np.random.seed(None)
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
INTERPOLATION_METHODS = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos4': cv2.INTER_LANCZOS4
    }

class CocoProcessor:
    def __init__(self, annotation_path):
        self.coco = COCO(annotation_path)
        self.img_dir = os.path.dirname(annotation_path)  # Assuming images are in the same directory as the JSON file

    def read_image(self, image_id):
        img_info = self.coco.loadImgs([image_id])[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        return cv2.imread(image_path)

    def crop_annotations(self, image, ann_ids):
        cropped_regions = []
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)[0]
            bbox = [int(i) for i in ann['bbox']]  # Convert bbox to int
            x, y, w, h = bbox
            cropped_regions.append(image[y:y+h, x:x+w])
        return cropped_regions

    @staticmethod
    def resize_cropped(cropped_region, resize_value, interpolation=cv2.INTER_LINEAR):
        """
        Resize a cropped region based on provided value.
        If tuple is provided, it resizes to the tuple dimensions.
        If float is provided, it scales the cropped region.
        """
        if isinstance(resize_value, tuple):
            return cv2.resize(cropped_region, resize_value, interpolation=interpolation)
        elif isinstance(resize_value, float) or isinstance(resize_value, int):
            h, w = cropped_region.shape[:2]
            return cv2.resize(cropped_region, (int(w * resize_value), int(h * resize_value)), interpolation=interpolation)
        else:
            raise ValueError("Unsupported resize value. Provide a tuple or a float.")

    def get_image_annotations(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        return self.coco.loadAnns(ann_ids)

    def process_image(self, image_id, resize_value=None, interpolation=cv2.INTER_LINEAR, low_res=False):
        image = self.read_image(image_id)
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        cropped_regions = self.crop_annotations(image, ann_ids)

        if resize_value:
            for i, region in enumerate(cropped_regions):
                original_shape = region.shape[:2]
                cropped_regions[i] = self.resize_cropped(region, resize_value, interpolation)
                if low_res:
                    cropped_regions[i] = self.resize_cropped(cropped_regions[i], original_shape, interpolation)

        return cropped_regions

class ImageFolderProcessor:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def read_image(self, image_file):
        image_path = os.path.join(self.image_folder, image_file)
        return cv2.imread(image_path)

    def random_crop(self, image, crop_size, crop_center = False):
        h, w, _ = image.shape
        if crop_center:
            x = w // 2 - crop_size[0] // 2
            y = h // 2 - crop_size[1] // 2
        else:
            x = np.random.randint(0, w - crop_size[0] + 1)
            y = np.random.randint(0, h - crop_size[1] + 1)
        return image[y:y+crop_size[1], x:x+crop_size[0]]

    def process_image(self, image_file, crop_size=None, resize_value=None, interpolation=cv2.INTER_LINEAR, low_res=False):
        image = self.read_image(image_file)
        cropped = self.random_crop(image, crop_size) if crop_size else image

        if resize_value:
            original_shape = cropped.shape[:2]
            cropped_gt = cropped
            cropped_rs = CocoProcessor.resize_cropped(cropped, resize_value, interpolation)
            if low_res:
                cropped_rs = CocoProcessor.resize_cropped(cropped_rs, original_shape, interpolation)

        return cropped_gt, cropped_rs
    
def parse_args():
    parser = argparse.ArgumentParser(description='Randomly crop COCO-formatted datasets to create pairs of SR images')
    parser.add_argument('source', type=str, help='Path to COCO json annotation file')
    parser.add_argument('--resize_value', type=str, default=None, help='Resize value for cropped annotations. Can be a tuple or a float.', nargs="+")
    parser.add_argument('--interpolation', type=str, default='linear', choices=['nearest', 'linear', 'cubic', 'area', 'lanczos4'], help='Interpolation method for resizing.')
    parser.add_argument('--low-res', action='store_true', dest='low_res', help='If specified, after resizing, resize the image back to its original shape with the same interpolation.')
    parser.add_argument('--folder-mode', action='store_true', help='Specify this flag if the source is an image folder instead of a COCO json annotation file.')
    parser.add_argument('--output-folder', type=str, help='Path to the output folder')
    
    args = parser.parse_args()
    return args

def main(args):
    scale = None
    interpolation = INTERPOLATION_METHODS[args.interpolation]
    if args.resize_value:
        if len(args.resize_value) == 1:
            scale = float(args.resize_value[0])
        elif len(args.resize_value) == 2:
            scale = (int(args.resize_value[0]), int(args.resize_value[1]))
        else:
            raise ValueError("The given resize value are wrong.")


    if args.folder_mode:
        processor = ImageFolderProcessor(args.source)
        for img_file in processor.image_files:
            img_base = os.path.basename(img_file).rsplit('.', 1)[0]
            img_ext = os.path.basename(img_file).rsplit('.', 1)[1]
        
            cropped_gt, cropped_rs = processor.process_image(img_file, resize_value=scale,
                                                             interpolation=interpolation,
                                                             low_res=args.low_res)
            cv2.imwrite(os.path.join(args.output_folder,f'gt_{img_base}.{img_ext}'), cropped_gt)
            cv2.imwrite(os.path.join(args.output_folder,f'{args.interpolation}_{img_base}.{img_ext}'), cropped_rs)

            # You can then use/save the cropped image as required
    else:
        processor = CocoProcessor(args.source)
        image_ids = processor.coco.getImgIds()
        
        for img_id in image_ids:
            cropped_gt = processor.process_image(img_id, scale, interpolation, args.low_res)

if __name__ == '__main__':
    args = parse_args()
    main(args)


