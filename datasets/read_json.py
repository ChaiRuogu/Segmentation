from datasets.config import *
from configs.config import *
import json
import cv2
import numpy as np

def read_json_and_stack_to_3d(input_path):
    with open(input_path) as f:
        data = json.load(f)

    img_list, mask_list, filename_list = [], [], []
    for condition, _ in data["data"].items():
        for sequence in data["data"][condition]:
            sequence_num = len(sequence)
            for j in range(sequence_num):
                img_3d, mask_3d = [], []
                filename = sequence[j]['base_name']
                for k in range(-1, 2):
                    index = min(max(j + k, 0), sequence_num - 1)
                    img = cv2.imread(sequence[index]['image_path'], 0)
                    mask = cv2.imread(sequence[index]['mask_path'], -1)
                    mask_single = np.zeros(mask.shape[:2], dtype=np.uint8)
                    if mask.ndim > 2:
                        for color, target_value in MAPPING.items():
                            valid = np.all(mask == np.array(color), axis=-1)
                            mask_single[valid] = target_value
                    img_3d.append(img)
                    mask_3d.append(mask_single)
                img_list.append(np.stack(img_3d, -1))
                mask_list.append(np.stack(mask_3d, -1))
                filename_list.append(filename)

    return np.stack(img_list, 0), np.stack(mask_list, 0), filename_list

if __name__ == '__main__':
    imgs, masks, filenames = read_json_and_stack_to_3d(TRAIN_JSON_PATH)
    print(imgs.shape)
    print(masks.shape)



