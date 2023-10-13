#################################################################
# This script prepares an input folder for training a SR model
# as defined by the ESRGAN repo. 
#################################################################
import argparse
import cv2
import numpy as np
import os
import sys
from basicsr.utils import scandir
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import glob

def generate_multiscale_dataset(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(args.output, f'{basename}T{idx}.png'))

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f'{basename}T{idx+1}.png'))


def extract_subimages_main(args):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size
            and longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are GT folder and LQ folder to be processed for DIV2K dataset.
        After process, each sub_folder should have the same number of subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    # scan all images
    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info
    
    
def generate_meta_info(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.meta_input, args.meta_root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in img_paths:
            status = True
            if args.check:
                # read the image once for check, as some images may have errors
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f'Read {img_path} error: {error}')
                    status = False
                if img is None:
                    status = False
                    print(f'Img is None: {img_path}')
            if status:
                # get the relative path
                img_name = os.path.relpath(img_path, root)
                print(img_name)
                txt_file.write(f'{img_name}\n')

def generate_paired_meta_info(args):
    assert len(args.meta_input) == 2, 'Input folder should have two elements: gt folder and lq folder'
    assert len(args.meta_root) == 2, 'Root path should have two elements: root for gt folder and lq folder'
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    for i in range(2):
        if args.input[i].endswith('/'):
            args.input[i] = args.input[i][:-1]
        if args.meta_root[i] is None:
            args.meta_root[i] = os.path.dirname(args.meta_input[i])

    txt_file = open(args.meta_info, 'w')
    # sca images
    img_paths_gt = sorted(glob.glob(os.path.join(args.meta_input[0], '*')))
    img_paths_lq = sorted(glob.glob(os.path.join(args.meta_input[1], '*')))

    assert len(img_paths_gt) == len(img_paths_lq), ('GT folder and LQ folder should have the same length, but got '
                                                    f'{len(img_paths_gt)} and {len(img_paths_lq)}.')

    for img_path_gt, img_path_lq in zip(img_paths_gt, img_paths_lq):
        # get the relative paths
        img_name_gt = os.path.relpath(img_path_gt, args.meta_root[0])
        img_name_lq = os.path.relpath(img_path_lq, args.meta_root[1])
        print(f'{img_name_gt}, {img_name_lq}')
        txt_file.write(f'{img_name_gt}, {img_name_lq}\n')

if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='datasets/DF2K/DF2K_multiscale', help='Output folder')
    parser.add_argument('--process', type=str,  choices=['multiscale','subimages','meta'], required=True)
    parser.add_argument('--crop_size', type=int, default=480, help='Crop size')
    parser.add_argument('--step', type=int, default=240, help='Step for overlapped sliding window')
    parser.add_argument(
        '--thresh_size',
        type=int,
        default=0,
        help='Threshold size. Patches whose size is lower than thresh_size will be dropped.')
    parser.add_argument('--n_thread', type=int, default=20, help='Thread number.')
    parser.add_argument('--compression_level', type=int, default=3, help='Compression level')
    parser.add_argument(
    '--meta_input',
    nargs='+',
    default=['datasets/DF2K/DF2K_HR', 'datasets/DF2K/DF2K_multiscale'],
    help='Input folder, can be a list')
    parser.add_argument(
        '--meta_root',
        nargs='+',
        default=['datasets/DF2K', 'datasets/DF2K'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt',
        help='txt path for meta info')
    parser.add_argument(
        '--paired_meta_info',
        action='store_true',
        help="Whether create meta information with paired (HQ,LQ) dataset"
    )
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()
    # coco_obj = None
    # if args.coco_input_json:
    #     coco_obj = COCO(args.coco_input_json)

    if 'multiscale' == args.process:
        generate_multiscale_dataset(args)
    elif 'subimages' == args.process:
        extract_subimages_main(args)
    elif 'meta' == args.process:
        if args.paired_meta_info:
            generate_paired_meta_info(args)
        else:
            generate_meta_info(args)
    else:
        raise ValueError('Undefined choice of a process')