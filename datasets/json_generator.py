from datasets.config import *
import os
import json
import re
from collections import defaultdict

def generate_dataset_json(image_dir, mask_dir, output_json_path):
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    pattern = re.compile(r'^(4\.\dV|pristine)-(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)
    grouped_data = defaultdict(list)

    for img_file in image_files:
        match = pattern.match(img_file)
        if match:
            prefix, number = match.group(1), int(match.group(2))
            mask_file = f"{prefix}-{number:03d}.png"
            if mask_file in mask_files:
                grouped_data[prefix].append({
                    "number": number,
                    "image_path": os.path.join(image_dir, img_file),
                    "mask_path": os.path.join(mask_dir, mask_file),
                    "base_name": f"{prefix}-{number:03d}"
                })

    for prefix in grouped_data:
        grouped_data[prefix].sort(key=lambda x: x["number"])

    dataset_info = {"categories": list(grouped_data.keys()), "data": {}}
    for prefix, items in grouped_data.items():
        dataset_info["data"][prefix] = []
        current_sequence = []
        prev_num = None
        for item in items:
            num = item["number"]
            if prev_num is None or num == prev_num + 1:
                current_sequence.append(item)
            else:
                dataset_info["data"][prefix].append(current_sequence)
                current_sequence = [item]
            prev_num = num
        if current_sequence:
            dataset_info["data"][prefix].append(current_sequence)

    with open(output_json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"JSON已生成至: {output_json_path}")

if __name__ == "__main__":
    generate_dataset_json(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TRAIN_JSON_PATH)
    generate_dataset_json(TEST_IMAGE_DIR, TEST_MASK_DIR, TEST_JSON_PATH)