import os
import torch
from datetime import datetime

from models.unet import UNet

def save_model_checkpoint(model, 
                         save_dir="checkpoint", 
                         filename=None, 
                         verbose=True):

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    default_name = f"{model_name}_params_{timestamp}.pth"
    filename = filename or default_name

    save_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), save_path)

    if verbose:
        print(f"模型参数已成功保存至：{save_path}")
        print(f"├─ 包含参数层数：{len(model.state_dict())}")
        print(f"└─ 总参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    return save_path

def inspect_model_layers(model, print_detail=False):
    state_dict = model.state_dict()
    param_info = {}
    
    print(f"\n模型结构分析：{model.__class__.__name__}")
    print(f"├─ 总层数：{len(state_dict)}")
    print(f"└─ 可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if print_detail:
        print("\n详细参数结构：")
        for i, (name, param) in enumerate(state_dict.items()):
            shape_str = "x".join(map(str, param.shape))
            dtype_str = str(param.dtype).replace("torch.", "") 
            
            param_info[name] = {
                "shape": param.shape,
                "dtype": param.dtype,
                "requires_grad": param.requires_grad
            }

            print(f"层 {i+1:03d} | {name:<40} | 形状：{shape_str:<16} | 类型：{dtype_str:<8} | 可训练：{param.requires_grad}")
    
    return param_info

if __name__ == "__main__":

    model = UNet(n_channels=3, n_classes=4)

    param_info = inspect_model_layers(model, print_detail=True)

    saved_path = save_model_checkpoint(model)

    loaded_state_dict = torch.load(saved_path)
    print("\n加载验证结果：")
    print(f"匹配参数层数：{len(loaded_state_dict)}/{len(model.state_dict())}")
    print(f"首层参数示例：{next(iter(loaded_state_dict.items()))[1].shape}")