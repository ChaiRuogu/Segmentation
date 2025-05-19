import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=torch.FloatTensor([0.03108941, 0.02882475, 0.11244715, 0.82763869]), gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.cuda()
        self.gamma = gamma

    def forward(self, pred, target): # pred:(N,num_class,H,W), target:(N,H,W)
        # 计算交叉熵损失
        # pdb.set_trace()
        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)  # (N, H, W)
        # 计算 Focal Loss
        pt = torch.exp(-ce_loss)  # 模型对真实类别的预测概率 (N,H,W)
        focal_loss = (self.alpha[target] * (1 - pt) ** self.gamma * ce_loss).mean()  # 加权平均
        return focal_loss

    
def dice_coefficient(pred, target):
    dice_scores = []
    for cls in range(NUM_CLASSES):
        p = (pred   == cls).astype(np.uint8)
        g = (target == cls).astype(np.uint8)

        inter  = (p & g).sum()
        denom  = p.sum() + g.sum()

        if denom == 0:           # 该类别在两图里都不存在
            dice = 1.0           # 视为完美匹配
        else:
            dice = (2.0 * inter) / denom
        dice_scores.append(float(dice))

    return dice_scores           # 保证每个 dice ∈ [0,1]

def pred_to_color(pred):
    
	pred_color = np.zeros((1600, 1600, 3), dtype = np.uint8)
	for class_id in range(NUM_CLASSES):
		pred_color[pred == class_id] = np.array(COLORS[class_id]).astype(np.uint8)
	return pred_color

def color_to_pred(mask):
	pred = np.zeros_like(mask, dtype=np.uint8)
	for color, classes in MAPPING.items():
		# pred[mask == color] = classes # 如果classes为数组则这条是错误代码,如果是标量则为正确
		pred[np.all(mask == color, -1)] = classes
	return pred

def process_logits_to_mask(logits):
    """
    将 logits 张量从形状 (N, class*depth, H, W) 转换为 (depth, H, W)
    
    参数:
        logits: 输入的 logits 张量，形状为 (N, class*depth, H, W)
    
    返回:
        processed_logits: 处理后的张量，形状为 (depth, H, W)的np数组
    """
    mask_list = []
    for i in range(DEPTH):
        logit_i = logits[:, i:(i+1)*4,: ,:] # (N, class, H, W)
        argmax_logit_i = torch.argmax(logit_i, dim=1).squeeze(0).cpu().numpy() # (H,W)
        mask_list.append(argmax_logit_i)
    mask = np.stack(mask.list)
    return mask
