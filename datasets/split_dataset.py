import os
import shutil
import random
from datasets.config import *

def split_dataset(test_ratio=TEST_RATIO, seed=42):
    random.seed(seed)

    for path in [TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_IMAGE_DIR, TEST_MASK_DIR]:
        os.makedirs(path, exist_ok=True)

    images = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.endswith(('png', 'jpg', 'jpeg'))])
    random.shuffle(images)

    num_test = int(len(images) * test_ratio)
    test_images = images[:num_test]
    train_images = images[num_test:]

    for image_set, img_dir, mask_dir in [(train_images, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR),
                                         (test_images, TEST_IMAGE_DIR, TEST_MASK_DIR)]:
        for image_name in image_set:
            shutil.copy(os.path.join(ORIGINAL_IMAGE_DIR, image_name), os.path.join(img_dir, image_name))
            shutil.copy(os.path.join(ORIGINAL_MASK_DIR, image_name), os.path.join(mask_dir, image_name))

    print(f"数据集分割完成: 训练集({len(train_images)}张)，测试集({len(test_images)}张)")

if __name__ == '__main__':
    split_dataset()