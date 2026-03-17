from tqdm import tqdm
import numpy as np
import math
from typing import Iterable
from copy import deepcopy
import os
from PIL import Image
import argparse

class Patches:
    def __init__(self, imgs, EM_indices):
        self.imgs = np.array(imgs)
        self.old_imgs = None
        self._EM_indices = EM_indices

    def update(self, imgs, shift_indices):
        if not isinstance(shift_indices, Iterable):
            raise TypeError('param "shifted_indices is not iteratable."')

        if self.is_updated():
            raise ValueError('Patches already updated. please .reset() before update.')

        if (len(imgs.shape) == 3 and len(shift_indices) == 1) or (
            len(imgs.shape) == 4 and len(shift_indices) == imgs.shape[0]
        ):
            self.old_imgs = deepcopy(self.imgs)
            self.imgs[shift_indices] = imgs
        else:
            raise ValueError('Image shape and index not Matched.')

    def is_updated(self):
        return True if self.old_imgs is not None else False

    def reset(self):
        if self.is_updated():
            self.imgs = self.old_imgs
            self.old_imgs = None

class EMPatches(object):
    def __init__(self):
        pass

    def extract_patches(self, img, patchsize, overlap=None, stride=None):
        height = img.shape[0]
        width = img.shape[1]
        maxWindowSize = patchsize
        windowSizeX = maxWindowSize
        windowSizeY = maxWindowSize
        windowSizeX = min(windowSizeX, width)
        windowSizeY = min(windowSizeY, height)

        if stride is not None:
            stepSizeX = stride
            stepSizeY = stride
        elif overlap is not None:
            overlapPercent = overlap
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)

            windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
            windowOverlapY = int(math.floor(windowSizeY * overlapPercent))

            stepSizeX = windowSizeX - windowOverlapX
            stepSizeY = windowSizeY - windowOverlapY
        else:
            stepSizeX = 1
            stepSizeY = 1

        lastX = width - windowSizeX
        lastY = height - windowSizeY
        xOffsets = list(range(0, lastX + 1, stepSizeX))
        yOffsets = list(range(0, lastY + 1, stepSizeY))

        if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            xOffsets.append(lastX)
        if len(yOffsets) == 0 or yOffsets[-1] != lastY:
            yOffsets.append(lastY)

        img_patches = []
        indices = []

        for xOffset in xOffsets:
            for yOffset in yOffsets:
                if len(img.shape) >= 3:
                    img_patches.append(
                        img[(slice(yOffset, yOffset + windowSizeY, None), slice(xOffset, xOffset + windowSizeX, None))]
                    )
                else:
                    img_patches.append(
                        img[(slice(yOffset, yOffset + windowSizeY), slice(xOffset, xOffset + windowSizeX))]
                    )
                indices.append((yOffset, yOffset + windowSizeY, xOffset, xOffset + windowSizeX))

        return img_patches, indices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_txt', type=str, help="path of dataset txt file having the path to <image> <depth> <mask>")
    parser.add_argument('--save_dir', type=str, default="./dataset/train", help="path to dir where dataset will be saved")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    emp = EMPatches()

    home_dir = os.path.abspath(args.save_dir)
    os.makedirs("./dataset_paths", exist_ok=True)
    
    # Define output file for the NEW patch-based dataset
    output_txt_path = os.path.abspath("./dataset_paths/train_extended.txt")

    with open(output_txt_path, 'w') as f2:
        with open(args.dataset_txt, "r") as f1:
            # Now reads 3 values per line
            for line in tqdm(f1):
                try:
                    image_path, target_path, mask_path = line.strip().split()
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                # Load Images
                img = Image.open(image_path)
                img = np.array(img)
                
                mask = Image.open(mask_path)
                mask = np.array(mask)
                
                depth = np.load(target_path)

                # Extract Patches
                img_patches, _ = emp.extract_patches(img, patchsize=1400, overlap=0.2)
                mask_patches, _ = emp.extract_patches(mask, patchsize=1400, overlap=0.2)
                depth_patches, _ = emp.extract_patches(depth, patchsize=1400, overlap=0.2)

                # Prepare Directory
                # Using folder name (e.g., 'Bathroom') to organize output
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(image_path))) # Adjusts based on structure
                if folder_name == "raw_training_data": # Fallback if structure differs
                     folder_name = os.path.basename(os.path.dirname(image_path))
                     
                dir_path = os.path.join(home_dir, folder_name)
                os.makedirs(dir_path, exist_ok=True)

                # Get base filename (e.g., 'im0') to prefix files
                base_filename = os.path.splitext(os.path.basename(image_path))[0]

                for i, (p_img, p_mask, p_depth) in enumerate(zip(img_patches, mask_patches, depth_patches)):
                    # Construct Paths
                    # We add base_filename so im0 patches don't overwrite im1 patches
                    path_img_save = os.path.join(dir_path, f"{base_filename}_{i}_img.npy")
                    path_depth_save = os.path.join(dir_path, f"{base_filename}_{i}_depth.npy")
                    path_mask_save = os.path.join(dir_path, f"{base_filename}_{i}_mask.npy")

                    if p_depth.max() != p_depth.min():
                        if np.sum(p_mask) > 0:
                            np.save(path_img_save, p_img)
                            np.save(path_depth_save, p_depth)
                            np.save(path_mask_save, p_mask)
                            
                            # Write formatted line: IMG DEPTH MASK
                            f2.write(f"{path_img_save} {path_depth_save} {path_mask_save}\n")