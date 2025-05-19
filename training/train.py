import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNet
from datasets.dataset3d import SegmentationDataset, default_transform
from datasets.config import *
from configs.config import *
from utils.utils import FocalLoss, dice_coefficient, pred_to_color
from datasets.read_json import read_json_and_stack_to_3d


class Runner:
    def __init__(self, crop_size=512, batch_size=4, lr=1e-3, device=None):
        # 设备
        self.device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
        )
        # 模型
        self.model = UNet(n_channels=DEPTH, n_classes=NUM_CLASSES).to(self.device)

        # 数据集
        self.train_dataset = SegmentationDataset(datatype='train', crop_size=crop_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.test_dataset = SegmentationDataset(datatype='test', crop_size=crop_size, transform=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        # 损失 & 优化器
        self.criterion = FocalLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 日志 & 检查点
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)
        self.writer = SummaryWriter(LOG_PATH)
        self.start_epoch = 0

    # -------------------- 评估 --------------------
    def evaluate(self, num_samples=50):
        """随机抽 num_samples 个 patch，返回各类别平均 Dice"""
        self.model.eval()
        dice_scores = []
        for _ in range(num_samples):
            idx = np.random.randint(0, len(self.test_dataset))
            image, mask = self.test_dataset[idx]          # image:(DEPTH,512,512)
            image = image.unsqueeze(0).to(self.device)    # (1,DEPTH,H,W)
            mask = mask.numpy()

            with torch.no_grad():
                logits = self.model(image)                # (1,DEPTH*CLS,H,W)
                dices = []
                for i in range(DEPTH):
                    frame_logits = logits[:, i*NUM_CLASSES:(i+1)*NUM_CLASSES]
                    pred = torch.argmax(frame_logits, 1).squeeze(0).cpu().numpy()
                    dices.append(dice_coefficient(pred, mask[i]))
                dice_scores.append(np.mean(dices, 0))     # (CLS,)

        return np.mean(dice_scores, 0)                     # (CLS,)

    # -------------------- 训练循环 -----------------
    def train(self, num_epochs=10000):
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.model.train()
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                logits = self.model(images)               # (B,DEPTH*CLS,H,W)
                frame_losses = []
                for i in range(DEPTH):
                    logits_i = logits[:, i*NUM_CLASSES:(i+1)*NUM_CLASSES]
                    weight = 10 if i == 1 else 1          # 中间帧权重更高
                    frame_losses.append(self.criterion(logits_i, masks[:, i]) * weight)
                loss = sum(frame_losses)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # 评估 & 保存
            if (epoch + 1) % 20 == 0:
                dice = self.evaluate()
                for cls, d in enumerate(dice):
                    self.writer.add_scalar(f'Dice/class_{cls}', d, epoch)
            if (epoch + 1) % 1000 == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_PATH, f'unet_epoch_{epoch+1}.pth'))

        self.writer.close()


if __name__ == '__main__':
    runner = Runner()
    runner.train()
