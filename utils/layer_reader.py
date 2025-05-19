import torch
import argparse
from pathlib import Path
from typing import Set, List

def load_state_dict(model_path: str) -> dict:
    """智能加载模型state_dict"""
    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
    
    if isinstance(model_data, dict):
        if "model_state_dict" in model_data:
            return model_data["model_state_dict"]
        if "state_dict" in model_data:
            return model_data["state_dict"]
        return model_data
    if isinstance(model_data, torch.nn.Module):
        return model_data.state_dict()
    raise ValueError("无法识别的模型文件格式")

def print_analysis_report(state_dict: dict, model_path: str, save_txt: bool):
    """生成单模型分析报告"""
    layer_names = list(state_dict.keys())
    
    print(f"\n[模型分析报告]")
    print(f"文件路径: {model_path}")
    print(f"参数总数: {len(layer_names)}")
    print("=" * 60)
    report_lines = []
    for i, name in enumerate(layer_names, 1):
        param_shape = tuple(state_dict[name].shape)
        line = f"{i:3d}. {name} - 形状: {param_shape}"
        print(line)
        report_lines.append(line)

    if save_txt:
        txt_path = Path(model_path).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"\n参数列表已保存至: {txt_path}")

def compare_models(state_dict1: dict, state_dict2: dict, 
                  model1_path: str, model2_path: str, 
                  save_txt: bool):
    """生成模型对比报告"""
    layers1 = set(state_dict1.keys())
    layers2 = set(state_dict2.keys())

    common = layers1 & layers2
    unique1 = layers1 - layers2
    unique2 = layers2 - layers1

    print("\n" + "="*60)
    print("[模型对比报告]")
    print(f"模型A: {model1_path} (参数层数: {len(layers1)})")
    print(f"模型B: {model2_path} (参数层数: {len(layers2)})")
    print("-"*60)
    
    def print_section(title: str, layers: Set[str]):
        print(f"\n[{title}] ({len(layers)} 个)")
        for i, layer in enumerate(sorted(layers), 1):
            print(f"{i:3d}. {layer}")

    print_section("共同参数层", common)
    print_section("模型A特有层", unique1)
    print_section("模型B特有层", unique2)

    if save_txt:
        base_name = f"{Path(model1_path).stem}_vs_{Path(model2_path).stem}"
        
        def save_layers(filename: str, layers: Set[str]):
            with open(f"{base_name}_{filename}.txt", "w") as f:
                f.write("\n".join(sorted(layers)))

        save_layers("common", common)
        save_layers("unique_A", unique1)
        save_layers("unique_B", unique2)
        print(f"\n对比结果已保存至: {base_name}_*.txt")

def main():
    parser = argparse.ArgumentParser(description='模型参数分析工具')
    parser.add_argument('model_paths', nargs='+', 
                       help='模型文件路径（1个分析/2个对比）')
    parser.add_argument('--save', action='store_true',
                       help='保存结果到txt文件')
    args = parser.parse_args()

    for path in args.model_paths:
        if not Path(path).exists():
            print(f"错误: 文件 {path} 不存在!")
            exit(1)

    try:
        if len(args.model_paths) == 1:
            state_dict = load_state_dict(args.model_paths[0])
            print_analysis_report(state_dict, args.model_paths[0], args.save)

        elif len(args.model_paths) == 2:
            sd1 = load_state_dict(args.model_paths[0])
            sd2 = load_state_dict(args.model_paths[1])
            compare_models(sd1, sd2, args.model_paths[0], args.model_paths[1], args.save)
        
        else:
            print("错误: 最多支持同时分析2个模型")
            exit(1)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        if isinstance(e, (KeyError, ValueError)):
            print("提示: 请检查模型文件格式是否正确")


if __name__ == "__main__":
    main()
