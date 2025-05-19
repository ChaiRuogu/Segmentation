import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL_IMAGE_DIR = os.path.join(BASE_DIR, 'original_data/images')
ORIGINAL_MASK_DIR = os.path.join(BASE_DIR, 'original_data/masks')

TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'datasets/train/image')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'datasets/train/mask')
TEST_IMAGE_DIR = os.path.join(BASE_DIR, 'datasets/test/image')
TEST_MASK_DIR = os.path.join(BASE_DIR, 'datasets/test/mask')

JSON_DIR = os.path.join(BASE_DIR, 'json')
TRAIN_JSON_PATH = os.path.join(JSON_DIR, 'dataset_train.json')
TEST_JSON_PATH = os.path.join(JSON_DIR, 'dataset_test.json')

TEST_RATIO = 0.2
