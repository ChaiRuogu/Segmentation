segmentation/                       # ── 根目录
│
├── environment.yml            # Conda 环境依赖（CUDA 12.1 + PyTorch 2.2 等）
├── PROJECT_STRUCTURE.md       # 本文件，描述项目结构
│
├── configs/                   # 全局配置（训练 & 推理共用）
│   └── config.py
│
├── datasets/                  # 数据集相关工具
│   ├── __init__.py
│   ├── config.py              # 数据路径统一管理
│   ├── dataset3d.py           # SegmentationDataset 类
│   ├── json_generator.py      # 生成 frame-seq JSON
│   ├── read_json.py           # 读取并堆叠 3D 图像
│   ├── preprocess.py          # 统计 mean/std、alpha 等
│   └── split_dataset.py       # 原始数据 ➜ train/test 分割
│
├── models/                    # 网络结构
│   ├── __init__.py
│   ├── unet.py                # 3D-UNet 主体
│   └── unet_parts.py          # UNet 子模块
│
├── training/                  # 训练 & 微调
│   ├── __init__.py
│   └── train.py               # Runner：训练 / 评估 / 保存
│
├── inference/                 # 推理、评估、可视化
│   ├── __init__.py
│   ├── config.py              # 推理阶段路径管理 (CKPT_FILE / PRED_DIR …)
│   ├── predictor.py           # 滑窗推理并输出彩色 mask
│   └── difference.py          # 预测 vs GT 差异图 + Dice 统计
│
├── utils/                     # 通用工具
│   ├── __init__.py
│   ├── utils.py               # FocalLoss、dice_coefficient 等
│   ├── checkpoint_get.py      # 选最新权重 / 命名工具
│   └── layer_reader.py        # 权重可视化 / Debug
│
├── checkpoints/               # 训练生成的 .pth 权重
├── logs/                      # TensorBoard 日志
├── pred_results/              # 推理输出彩色 mask（由 predictor.py 创建）
└── diff_results/              # 差异图输出（由 difference.py 创建）
