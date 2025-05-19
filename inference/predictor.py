"""
inference/predictor.py — 统一路径配置版
运行示例:  python -m inference.predictor
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from models.unet import UNet
from configs.config import NUM_CLASSES, DEPTH, MAPPING
from datasets.config import TRAIN_JSON_PATH, TEST_JSON_PATH
from datasets.read_json import read_json_and_stack_to_3d
from datasets.json_generator import generate_dataset_json
from datasets.dataset3d import default_transform
from inference.config import CKPT_FILE, PRED_DIR, INFER_JSON

class Predictor:
    def __init__(self, ckpt_path=CKPT_FILE, device=None, crop_size=512):
        self.device = torch.device("cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu"))
        self.model  = UNet(n_channels=DEPTH, n_classes=NUM_CLASSES).to(self.device)
        self.crop_size = crop_size
        self._load_checkpoint(ckpt_path)
        self.model.eval()

    # -------------------------
    def _load_checkpoint(self, ck_path):
        if not os.path.isfile(ck_path):
            raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
        state = torch.load(ck_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        print(f"✅ Loaded checkpoint: {ck_path}")

    # -------------------------
    def _inference_volume(self, images):
        """
        images: (len_seq, H, W, DEPTH)，返回 (len_seq, H, W) mask
        """
        seq_len, H, W, _ = images.shape
        stride = self.crop_size // 2
        output = np.zeros((seq_len, H, W, NUM_CLASSES), dtype=np.float32)

        for i in tqdm(range(seq_len), desc="Infer sequence"):
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y0, x0 = y, x
                    y1, x1 = min(y+self.crop_size, H), min(x+self.crop_size, W)
                    if y1 - y0 < self.crop_size:
                        y0 = max(0, y1 - self.crop_size)
                    if x1 - x0 < self.crop_size:
                        x0 = max(0, x1 - self.crop_size)

                    patch = images[i, y0:y1, x0:x1]                   # (h,w,DEPTH)
                    patch_t = default_transform(patch).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.model(patch_t)                  # (1,DEPTH*CLS,h,w)
                        mid_logits = logits[:, NUM_CLASSES:2*NUM_CLASSES]
                        pred_mid = torch.argmax(mid_logits, 1).squeeze(0).cpu().numpy()

                    for cls in range(NUM_CLASSES):
                        output[i, y0:y1, x0:x1, cls] += (pred_mid == cls)

        return np.argmax(output, -1)                                  # (seq_len,H,W)

    # -------------------------
    def predict_json(self, json_path, save_dir=PRED_DIR):
        os.makedirs(save_dir, exist_ok=True)
        imgs, _, filenames = read_json_and_stack_to_3d(json_path)     # imgs:(N,H,W,DEPTH)
        preds = self._inference_volume(imgs)                          # (N,H,W)

        for i, name in enumerate(tqdm(filenames, desc="Save masks")):
            pred_color = np.zeros((*preds.shape[1:], 3), np.uint8)
            for color, cls in MAPPING.items():
                pred_color[preds[i] == cls] = color
            cv.imwrite(os.path.join(save_dir, f"{name}.png"), pred_color)

        print(f"🎉 All predictions saved to {save_dir}")

# ----------------------------- main -----------------------------
if __name__ == '__main__':
    # 若需要重新生成 json，可取消注释
    # generate_dataset_json('datasets/test/image', 'datasets/test/mask', INFER_JSON)

    predictor = Predictor(ckpt_path=CKPT_FILE)
    predictor.predict_json(INFER_JSON)
