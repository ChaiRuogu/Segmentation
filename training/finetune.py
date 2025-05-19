import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.unet import UNet
from datasets.dataset3d import SegmentationDataset
from utils.utils import FocalLoss, dice_coefficient, pred_to_color
from configs.config import *

PRETRAINED_PATH = "unet3d_pretrained/checkpoint/MODEL.pth"
NUM_EPOCHS = 20000
BASE_LR = 8e-6
FT_LR = 1e-3
BATCH_SIZE = 4
CROP_SIZE = 512
NUM_VAL_SAMPLES = 100

class FineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=DEPTH, n_classes=NUM_CLASSES).to(self.device)
        self._load_pretrained()
        self.optimizer = optim.Adam(self.model.parameters(), lr=FT_LR)
        self.criterion = FocalLoss()

        self.train_dataset = SegmentationDataset(TRAIN_DATA_PATH + 'image/', TRAIN_DATA_PATH + 'mask/', crop_size=CROP_SIZE, datatype='train')
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.val_dataset = SegmentationDataset(TEST_DATA_PATH + 'image/', TEST_DATA_PATH + 'mask/', crop_size=CROP_SIZE, datatype='test', transform=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        self.writer = SummaryWriter(log_dir=LOG_PATH)

    def _load_pretrained(self):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=self.device)
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('outc')}
        self.model.load_state_dict(state_dict, strict=False)

    def validate(self, epoch):
        self.model.eval()
        dice_scores = []

        for idx in range(NUM_VAL_SAMPLES):
            image, mask = self.val_dataset[idx]
            image = image.unsqueeze(0).to(self.device)
            mask = mask.numpy()

            with torch.no_grad():
                logits = self.model(image)
                mid_logits = logits[:, 4:8, :, :]
                pred_mid = torch.argmax(mid_logits, dim=1).squeeze(0).cpu().numpy()
                dice_mid = dice_coefficient(pred_mid, mask[1])
                dice_scores.append(dice_mid)

        dice_scores = np.array(dice_scores)
        mean_dice_per_class = np.mean(dice_scores, axis=0)
        overall_dice = np.mean(mean_dice_per_class)

        self.writer.add_scalar('Dice/overall', overall_dice, epoch)
        for cls in range(NUM_CLASSES):
            self.writer.add_scalar(f'Dice/class_{cls}', mean_dice_per_class[cls], epoch)

        print(f"\n=== 验证结果 Epoch {epoch+1} ===")
        for cls in range(NUM_CLASSES):
            print(f"类别 {cls}: {mean_dice_per_class[cls]:.4f}")
        print(f"平均Dice: {overall_dice:.4f}")

        return overall_dice

    def run(self):
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = sum(
                    self.criterion(outputs[:, i*NUM_CLASSES:(i+1)*NUM_CLASSES], masks[:, i]) * (10 if i == 1 else 1)
                    for i in range(DEPTH)
                )

                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {loss.item():.4f}")
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            if (epoch + 1) % 1000 == 0:
                self.validate(epoch)
                torch.save(self.model.state_dict(), f"{CHECKPOINT_PATH}/ft_epoch_{epoch}.pth")

        self.writer.close()

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    tuner = FineTuner()
    tuner.run()
