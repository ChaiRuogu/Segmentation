# inference/config.py
import os
from datasets.config import JSON_DIR, TEST_JSON_PATH         # 直接复用 datasets 中已存在的常量
from datasets.config import TEST_MASK_DIR

# 推理用模型权重
CKPT_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
CKPT_FILE = os.path.join(CKPT_DIR, "unet_epoch_1000.pth")      # 你可换成最新权重

# 预测输出目录
PRED_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pred_results")
os.makedirs(PRED_DIR, exist_ok=True)

# 待预测 json（此示例直接用测试集 json）
INFER_JSON = TEST_JSON_PATH   

GT_MASK_DIR = TEST_MASK_DIR               
DIFF_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diff_results")
os.makedirs(DIFF_DIR, exist_ok=True)