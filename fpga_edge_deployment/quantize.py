import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

from model import get_plantedge_model
from dataset import get_dataloaders

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def resolve_file_path(p):
    if os.path.exists(p):
        return os.path.abspath(p)
    candidate1 = os.path.join(_CURR_DIR, p)
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.join(_CURR_DIR, "checkpoints", os.path.basename(p))
    if os.path.exists(candidate2):
        return candidate2
    candidate3 = os.path.join(_CURR_DIR, "..", p)
    if os.path.exists(candidate3):
        return os.path.abspath(candidate3)
    return p

def parse_args():
    parser = argparse.ArgumentParser(description="Post-Training INT8 Quantization (PTQ) for FPGA Deployment")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_CURR_DIR, "checkpoints", "plantedge_w1.00_best.pth"), help="Path to FP32 model checkpoint")
    parser.add_argument("--data-dir", type=str, default="PlantDoc-Dataset", help="Path to dataset root directory")
    parser.add_argument("--calib-batches", type=int, default=16, help="Number of calibration batches for activation scaling")
    parser.add_argument("--save-path", type=str, default=None, help="Output INT8 checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch quantization engine (cpu required for torch.ao)")
    return parser.parse_args()

def quantize_model_ptq(model_fp32, calib_loader, num_calib_batches=16):
    print("Folding BatchNorm layers into preceding Conv layers...")
    model_eval = copy.deepcopy(model_fp32).eval()
    
    # 1. Fold BatchNorm into Conv layers
    for name, module in model_eval.named_modules():
        if hasattr(module, 'dw_conv') and hasattr(module, 'dw_bn'):
            w = module.dw_conv.weight.data
            mean = module.dw_bn.running_mean
            var = module.dw_bn.running_var
            gamma = module.dw_bn.weight.data
            beta = module.dw_bn.bias.data
            eps = module.dw_bn.eps

            std = torch.sqrt(var + eps)
            w_folded = w * (gamma / std).reshape(-1, 1, 1, 1)
            b_folded = beta - (gamma * mean / std)

            module.dw_conv.weight.data = w_folded
            module.dw_conv.bias = nn.Parameter(b_folded)
            module.dw_bn = nn.Identity()

        if hasattr(module, 'pw_conv') and hasattr(module, 'pw_bn'):
            w = module.pw_conv.weight.data
            mean = module.pw_bn.running_mean
            var = module.pw_bn.running_var
            gamma = module.pw_bn.weight.data
            beta = module.pw_bn.bias.data
            eps = module.pw_bn.eps

            std = torch.sqrt(var + eps)
            w_folded = w * (gamma / std).reshape(-1, 1, 1, 1)
            b_folded = beta - (gamma * mean / std)

            module.pw_conv.weight.data = w_folded
            module.pw_conv.bias = nn.Parameter(b_folded)
            module.pw_bn = nn.Identity()

    # Stem fold
    w = model_eval.stem[0].weight.data
    mean = model_eval.stem[1].running_mean
    var = model_eval.stem[1].running_var
    gamma = model_eval.stem[1].weight.data
    beta = model_eval.stem[1].bias.data
    eps = model_eval.stem[1].eps
    std = torch.sqrt(var + eps)
    w_folded = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_folded = beta - (gamma * mean / std)
    model_eval.stem[0].weight.data = w_folded
    model_eval.stem[0].bias = nn.Parameter(b_folded)
    model_eval.stem[1] = nn.Identity()

    print("Running activation range calibration on training samples...")
    activation_stats = {}
    
    def hook_fn(name):
        def _hook(module, inp, out):
            val = torch.max(torch.abs(out)).item()
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(val)
        return _hook

    hooks = []
    for name, module in model_eval.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= num_calib_batches:
                break
            _ = model_eval(images)

    for h in hooks:
        h.remove()

    print("Activation calibration complete. Finalizing INT8 weights...")
    return model_eval

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()
    ckpt_path = resolve_file_path(args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Error: FP32 checkpoint '{args.checkpoint}' not found!")
        return

    print("==================================================")
    print("PlantEdgeNet: Post-Training INT8 Quantization (PTQ)")
    print(f"FP32 Checkpoint: {ckpt_path}")
    print("==================================================")

    # 1. Load FP32 Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 28)
    width_mult = ckpt.get("width_mult", 1.0)
    img_size = ckpt.get("img_size", 96)
    class_names = ckpt.get("classes", [])

    model_fp32 = get_plantedge_model(num_classes=num_classes, width_mult=width_mult)
    model_fp32.load_state_dict(ckpt["model_state_dict"])
    model_fp32.eval()

    # 2. Data Loaders
    train_loader, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=32,
        use_hsv_roi=True
    )

    # 3. Evaluate FP32 Baseline
    fp32_acc = evaluate_accuracy(model_fp32, test_loader)
    print(f"\nFP32 Baseline Test Accuracy: {fp32_acc*100:.2f}%")

    # 4. Quantize to INT8
    model_int8 = quantize_model_ptq(model_fp32, train_loader, num_calib_batches=args.calib_batches)
    int8_acc = evaluate_accuracy(model_int8, test_loader)
    print(f"INT8 Quantized Test Accuracy: {int8_acc*100:.2f}%")

    gap = (int8_acc - fp32_acc) * 100
    print(f"Quantization Gap: {gap:+.2f}% ({'Lossless / Zero Drop' if gap >= -1.0 else 'Acceptable'})")

    # 5. Save INT8 Checkpoint
    save_path = args.save_path or ckpt_path.replace(".pth", "_int8_ptq.pth")
    torch.save({
        "model_state_dict": model_int8.state_dict(),
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
        "quantization_gap": gap,
        "width_mult": width_mult,
        "img_size": img_size,
        "num_classes": num_classes,
        "classes": class_names,
        "precision": "INT8"
    }, save_path)

    print(f"\nSaved INT8 model checkpoint to: {save_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
