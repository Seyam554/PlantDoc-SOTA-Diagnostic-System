"""
PlantEdgeNet - a depthwise-separable CNN student for PlantDoc (28 classes)
designed to fit < 100K parameters and quantize cleanly to INT8 for an
Artix-7 XC7A200T dataflow accelerator (FINN / hls4ml / Vivado HLS).

Design rules (see research/RESEARCH_REPORT.md):
  * uint8 input (64 or 96 px); ImageNet normalization folded into the first
    conv at export time (train with normalized tensors as usual).
  * depthwise 3x3 + pointwise 1x1 blocks (MobileNet-style) for param economy.
  * optional Squeeze-Excite on the middle blocks (--se): +~9K params, +1-2 pts.
  * ReLU6 activations (piecewise-linear -> free in fixed point).
  * stride-2 convs for downsampling; GlobalAvgPool -> single FC(->28) head.
  * no cross-scale residual adds (keeps requantization simple).

Usage:
    python fpga/model_tiny.py --width 1.5 --se --img-size 96
"""

import argparse
import torch
import torch.nn as nn

NUM_CLASSES = 28
INPUT_SIZE = 64


def _c(ch, width_mult, divisor=8):
    """Round channel count to a multiple of `divisor` (hardware-friendly)."""
    ch = ch * width_mult
    new_ch = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return int(new_ch)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Channel attention. Params = 2*C*C/r. INT8-friendly (GAP + 2 FC + gate)."""

    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(4, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DWSep(nn.Module):
    """Depthwise 3x3 (stride s) -> pointwise 1x1 -> optional SE."""

    def __init__(self, in_ch, out_ch, s=1, se=False, se_r=16):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=s, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1)
        self.se = SqueezeExcite(out_ch, se_r) if se else nn.Identity()

    def forward(self, x):
        return self.se(self.pw(self.dw(x)))


class PlantEdgeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=1.0, dropout=0.1, se=False, se_r=16):
        super().__init__()
        w = width_mult
        c = lambda ch: _c(ch, w)
        # SE on the middle blocks only (cheapest params, most benefit)
        se_flags = [False, se, se, se, se, False]

        self.stem = ConvBNAct(3, c(16), k=3, s=2)                  # /2
        self.blocks = nn.Sequential(
            DWSep(c(16), c(24), s=1, se=se_flags[0], se_r=se_r),
            DWSep(c(24), c(32), s=2, se=se_flags[1], se_r=se_r),   # /4
            DWSep(c(32), c(48), s=1, se=se_flags[2], se_r=se_r),
            DWSep(c(48), c(64), s=2, se=se_flags[3], se_r=se_r),   # /8
            DWSep(c(64), c(96), s=1, se=se_flags[4], se_r=se_r),
            DWSep(c(96), c(128), s=2, se=se_flags[5], se_r=se_r),  # /16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c(128), num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        feats = self.blocks(x)
        x = self.pool(feats).flatten(1)
        logits = self.fc(self.drop(x))
        if return_features:
            return logits, feats
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model, size=INPUT_SIZE):
    """Rough MAC count via forward hooks on Conv2d / Linear."""
    macs = [0]
    handles = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oc, oh, ow = out.shape[1:]
            k = mod.kernel_size[0] * mod.kernel_size[1]
            macs[0] += oc * oh * ow * (mod.in_channels // mod.groups) * k
        elif isinstance(mod, nn.Linear):
            macs[0] += mod.in_features * mod.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(1, 3, size, size, device=dev))
    for h in handles:
        h.remove()
    return macs[0]


def build_model(width_mult=1.0, num_classes=NUM_CLASSES, dropout=0.1, se=False, se_r=16, assert_budget=True):
    model = PlantEdgeNet(num_classes=num_classes, width_mult=width_mult, dropout=dropout, se=se, se_r=se_r)
    n = count_params(model)
    if assert_budget:
        assert n < 100_000, f"PlantEdgeNet has {n} params (>= 100000). Lower --width or drop --se."
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--se", action="store_true")
    ap.add_argument("--img-size", type=int, default=INPUT_SIZE)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    m = build_model(width_mult=args.width, se=args.se, assert_budget=False)
    n = count_params(m)
    macs = count_macs(m, size=args.img_size)
    print(f"PlantEdgeNet(width_mult={args.width}, se={args.se}, img={args.img_size})")
    print(f"  parameters : {n:,}  ({'OK < 100K' if n < 100_000 else 'OVER BUDGET'})")
    print(f"  MACs/inf   : {macs:,}  (~{macs/1e6:.1f} M)")
    print(f"  INT8 weight bytes : {n:,} B  (fits XC7A200T BRAM ~1.6 MB)")
    if args.summary:
        print(m)
