"""
inference/difference.py — 受 inference.config 调控
生成差异图并计算 Dice
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm

from inference.config import PRED_DIR, GT_MASK_DIR, DIFF_DIR
from configs.config  import NUM_CLASSES, MAPPING

# ---------- Dice 计算 ----------
def dice_coeff(mask1, mask2, num_classes=NUM_CLASSES):
    dices = []
    for cls in range(num_classes):
        m1 = (mask1 == cls).astype(np.uint8)
        m2 = (mask2 == cls).astype(np.uint8)
        inter = (m1 & m2).sum()
        denom = m1.sum() + m2.sum()
        dices.append((2*inter) / (denom + 1e-7))
    return dices, float(np.mean(dices))

# ---------- 主流程 ----------
def compare_all():
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.png')]
    if not pred_files:
        raise RuntimeError(f"No prediction png found in {PRED_DIR}")

    mean_dices = []

    for file in tqdm(pred_files, desc="Diff & Dice"):
        pred_path = os.path.join(PRED_DIR, file)
        gt_path   = os.path.join(GT_MASK_DIR, file)

        if not os.path.isfile(gt_path):
            print(f"[Warn] GT not found: {gt_path}")
            continue

        pred_img = cv2.imread(pred_path)
        gt_img   = cv2.imread(gt_path)

        # 颜色→label
        pred = np.zeros(pred_img.shape[:2], np.uint8)
        gt   = np.zeros(gt_img.shape[:2],   np.uint8)
        for color, cls in MAPPING.items():
            pred[np.all(pred_img == color, -1)] = cls
            gt  [np.all(gt_img   == color, -1)] = cls

        # 差异图
        diff = np.zeros_like(pred_img)
        diff_mask = pred != gt
        diff[diff_mask] = gt_img[diff_mask]
        cv2.imwrite(os.path.join(DIFF_DIR, file), diff)

        # Dice
        dices, mean_dice = dice_coeff(pred, gt)
        mean_dices.append(mean_dice)

    if mean_dices:
        print(f"✅ 全部完成  平均Dice={np.mean(mean_dices):.4f}")


if __name__ == '__main__':
    compare_all()
