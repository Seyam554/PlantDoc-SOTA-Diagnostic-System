import math
import torch
import torch.nn as nn

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) for FPGA:
    Uses a 1D convolution with kernel size k=3 to capture cross-channel interactions
    with near-zero parameter overhead (~3-5 parameters per block!) and no dimensionality reduction.
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidualEdgeBlock(nn.Module):
    """
    FPGA-Optimized Inverted Residual Block with ECA Attention & ReLU6:
    1x1 Expansion Conv -> 3x3 Depthwise Conv -> ECA Channel Attention -> 1x1 Linear Projection + Residual Skip.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, use_eca=True):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        # 1. 1x1 Expansion (if expand_ratio != 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # 2. 3x3 Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # 3. Lightweight ECA Channel Attention
        if use_eca:
            layers.append(ECALayer(hidden_dim, k_size=3))

        # 4. 1x1 Linear Pointwise Projection (no non-linearity to preserve manifold)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Standard Depthwise Separable Conv: 3x3 DW + ReLU6 + 1x1 PW + ReLU6
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.ReLU6(inplace=True)
        
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


class PlantEdgeNet(nn.Module):
    """
    PlantEdgeNet V2: Ultra-Lightweight (<100K params) SOTA Edge Architecture
    Features Inverted Residual Bottlenecks, ECA Attention, and ReLU6 activations.
    """
    def __init__(self, num_classes=28, width_mult=1.0, arch_type="inverted_residual", in_channels=3, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.arch_type = arch_type

        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Base channel configuration (Strictly calibrated for ~99K-100K parameters)
        c_stem = _make_divisible(16 * width_mult)
        c1 = _make_divisible(24 * width_mult)
        c2 = _make_divisible(32 * width_mult)
        c3 = _make_divisible(48 * width_mult)
        c4 = _make_divisible(72 * width_mult)
        c5 = _make_divisible(96 * width_mult)
        c6 = _make_divisible(128 * width_mult)

        # 1. Stem: Conv 3x3 with stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU6(inplace=True)
        )

        # 2. Stage Blocks
        if arch_type == "inverted_residual":
            self.block1 = InvertedResidualEdgeBlock(c_stem, c1, stride=1, expand_ratio=1, use_eca=True)
            self.block2 = InvertedResidualEdgeBlock(c1, c2, stride=2, expand_ratio=2, use_eca=True)
            self.block3 = InvertedResidualEdgeBlock(c2, c3, stride=1, expand_ratio=2, use_eca=True)
            self.block4 = InvertedResidualEdgeBlock(c3, c4, stride=2, expand_ratio=2, use_eca=True)
            self.block5 = InvertedResidualEdgeBlock(c4, c5, stride=1, expand_ratio=2, use_eca=True)
            self.block6 = InvertedResidualEdgeBlock(c5, c6, stride=2, expand_ratio=2, use_eca=True)
        else:
            self.block1 = DepthwiseSeparableConv(c_stem, c1, stride=1)
            self.block2 = DepthwiseSeparableConv(c1, c2, stride=2)
            self.block3 = DepthwiseSeparableConv(c2, c3, stride=1)
            self.block4 = DepthwiseSeparableConv(c3, c4, stride=2)
            self.block5 = DepthwiseSeparableConv(c4, c5, stride=1)
            self.block6 = DepthwiseSeparableConv(c5, c6, stride=2)

        # 3. Global Pooling & Lightweight Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(c6, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_macs(self, input_size=(1, 3, 96, 96)):
        macs = 0
        def conv_hook(self, inp, out):
            nonlocal macs
            if isinstance(self, nn.Conv2d):
                output_channels, output_h, output_w = out.shape[1], out.shape[2], out.shape[3]
                kernel_h, kernel_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
                in_channels = self.in_channels // self.groups
                macs += output_channels * output_h * output_w * in_channels * kernel_h * kernel_w
            elif isinstance(self, nn.Conv1d):
                output_channels, output_l = out.shape[1], out.shape[2]
                k = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
                in_channels = self.in_channels // self.groups
                macs += output_channels * output_l * in_channels * k

        def linear_hook(self, inp, out):
            nonlocal macs
            macs += self.in_features * self.out_features

        hooks = []
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))

        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        dummy = torch.randn(*input_size, device=dev)
        _ = self(dummy)
        for h in hooks:
            h.remove()
        return macs

def get_plantedge_model(num_classes=28, width_mult=1.0, arch_type="inverted_residual"):
    return PlantEdgeNet(num_classes=num_classes, width_mult=width_mult, arch_type=arch_type)

if __name__ == "__main__":
    for wm in [0.75, 1.0, 1.25]:
        m = get_plantedge_model(num_classes=28, width_mult=wm, arch_type="inverted_residual")
        p = m.count_parameters()
        macs = m.count_macs((1, 3, 96, 96))
        print(f"PlantEdgeNet V2 (w={wm:.2f}): {p:,} params | {macs/1e6:.2f} M MACs | FPGA Compliant: {p < 100000}")
