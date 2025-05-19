from datasets.config import *
from configs.config import *
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2 as cv

def calculate_alpha():
    mask_folder = ORIGINAL_MASK_DIR
    class_counts = {}
    total_pixels = 0

    for mask_file in os.listdir(mask_folder):
        if not mask_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        mask_path = os.path.join(mask_folder, mask_file)
        mask_ori = cv.imread(mask_path, -1)
        mask = np.zeros((mask_ori.shape[0], mask_ori.shape[1]), dtype=np.uint8)
        for color, target_value in MAPPING.items():
            valid = np.where(np.all(mask_ori == np.array(color), axis=-1))
            class_counts[target_value] = len(valid[0])
        total_pixels += mask.shape[0] * mask.shape[1]

    classes = sorted(class_counts.keys())
    class_counts = [class_counts[c] for c in classes]
    class_freq = np.array(class_counts) / total_pixels

    alpha = 1.0 / class_freq
    alpha = alpha / alpha.sum()
    return alpha

def cal_mean_std():
    image_dir = TRAIN_IMAGE_DIR
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)

        mean += image_tensor.sum(dim=[1, 2])
        std += (image_tensor ** 2).sum(dim=[1, 2])
        num_pixels += image_tensor.shape[1] * image_tensor.shape[2]

    mean /= num_pixels
    std = (std / num_pixels - mean ** 2) ** 0.5

    print("Mean (R, G, B):", mean.tolist())
    print("Std (R, G, B):", std.tolist())