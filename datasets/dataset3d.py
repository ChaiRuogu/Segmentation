from datasets.config import *
from torch.utils.data import Dataset
from datasets.read_json import *
from datasets.json_generator import *
from torchvision import transforms
import numpy as np
import torch
import random
import cv2 as cv

class SegmentationDataset(Dataset):
    def __init__(self, datatype='train', crop_size=512, transform=True):
        if datatype == 'train':
            generate_dataset_json(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TRAIN_JSON_PATH)
            self.images, self.masks, _ = read_json_and_stack_to_3d(TRAIN_JSON_PATH)
        else:
            generate_dataset_json(TEST_IMAGE_DIR, TEST_MASK_DIR, TEST_JSON_PATH)
            self.images, self.masks, _ = read_json_and_stack_to_3d(TEST_JSON_PATH)

        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]

        h, w, _ = image.shape
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]

        if self.transform:
            angle = random.choice([0, 90, 180, 270])
            if angle:
                M = cv.getRotationMatrix2D((self.crop_size/2, self.crop_size/2), angle, 1)
                image = cv.warpAffine(image, M, (self.crop_size, self.crop_size))
                mask = cv.warpAffine(mask, M, (self.crop_size, self.crop_size), flags=cv.INTER_NEAREST)

            flip_code = random.choice([-1, 0, 1])
            if flip_code != -1:
                image = cv.flip(image, flip_code)
                mask = cv.flip(mask, flip_code)

        image = default_transform(image)
        mask = torch.from_numpy(mask).long().permute(2, 0, 1)

        return image, mask

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2055]*3, std=[0.0674]*3)
])
