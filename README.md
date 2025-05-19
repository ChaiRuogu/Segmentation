已思考 5 秒
> 适用仓库：`segmentation/`
> GPU 驱动 ≥ CUDA 12.1，已在 Windows/WSL & Ubuntu 测试通过。

---

## 1. 创建 Conda 环境

```bash
# 在项目根目录
conda env create -f environment.yml
conda activate segmentation
```

验证：

```bash
python - <<'PY'
import torch, numpy as np
print("CUDA =", torch.cuda.is_available(), "| torch", torch.__version__, "| numpy", np.__version__)
PY
```

应输出 `CUDA = True`。

---

## 2. 数据准备

### 2.1 原始数据放置

```
original_data/
├── images/   (*.jpg / *.png …)
└── masks/    (与 images 同名 *.png，RGB 标签色)
```

### 2.2 一键分割为 train / test

```bash
python -m datasets.split_dataset         # 默认 test_ratio=0.2
```

结果：

```
datasets/
├── train/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

### 2.3 可选：统计 mean/std 或类别 α

```bash
python -m datasets.preprocess cal_mean_std
python -m datasets.preprocess calculate_alpha
```

---

## 3. 训练

```bash
# 训练 + TensorBoard 日志
python -m training.train
tensorboard --logdir logs      # 另开终端查看曲线
```

* **权重**默认保存在 `checkpoints/`。
* 训练中每 500 epoch 评估一次，每 1000 epoch 保存一次模型。

如需修改超参，请编辑 `training/train.py` 顶部的 `Runner` 初始化参数或 `datasets/config.py` 中的路径/类别配置。

---

## 4. 推理

1. 将最佳模型重命名/放到
   `checkpoints/unet_epoch_XXXX.pth` 或修改 `inference/config.py → CKPT_FILE`。
2. 运行：

```bash
python -m inference.predictor        # 读取 TEST_JSON_PATH
```

* 彩色预测 mask 输出到 `pred_results/`
* 若需换数据，只需改 `inference/config.py → INFER_JSON`。

---

## 5. 评估差异 & Dice

```bash
python -m inference.difference
```

* 差异可视化图保存到 `diff_results/`
* 终端打印平均 Dice。

---

## 6. 常见问题 & 解决

| 问题                                       | 解决方式                                                             |
| ---------------------------------------- | ---------------------------------------------------------------- |
| **`torch.cuda.is_available()==False`**   | 1) 驱动 ≥ CUDA 12.1<br>2) 重新 `conda env create -f environment.yml` |
| **NumPy 2.0 兼容警告**                       | `pip install \"numpy<2\"`                                        |
| **ModuleNotFoundError: datasets.config** | 确保从仓库根目录运行：<br>`python -m training.train`                        |
| **找不到权重** (`FileNotFoundError`)          | 修改 `inference/config.py → CKPT_FILE` 指向正确 `.pth`                 |

---

## 7. 目录速览

详见 `PROJECT_STRUCTURE.md`，关键如下：

```
configs/        # 全局参数
datasets/       # 数据处理 & JSON
models/         # UNet
training/       # 训练脚本
inference/      # 推理 + 评估
utils/          # 通用工具
checkpoints/    # 权重
logs/           # TensorBoard
pred_results/   # 预测输出
diff_results/   # 差异图
```
