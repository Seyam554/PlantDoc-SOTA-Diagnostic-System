import torch
import torch.nn as nn
import math

class DepthwiseSeparableConv(nn.Module):
    """
    FPGA-friendly Depthwise Separable Convolution block:
    3x3 Depthwise Conv (per-channel) + BatchNorm + ReLU6 + 1x1 Pointwise Conv + BatchNorm + ReLU6.
    Uses ReLU6 for exact fixed-point piecewise linear clamping on FPGA DSP/LUTs.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        
        # Depthwise Conv
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.ReLU6(inplace=True)
        
        # Pointwise Conv
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
    PlantEdgeNet: Ultra-Lightweight, Sub-100K-Parameter INT8-Targeted CNN
    Designed for AMD Artix-7 XC7A200T FPGA deployment via Vivado / FINN / hls4ml.
    
    Key Hardware Constraints Respected:
    1. Parameter Count < 100,000 (fits entirely in on-chip 1.63 MB BRAM).
    2. Zero Float-only ops: No LayerNorm, No GELU, No Softmax in backbone.
    3. Fixed-point friendly activations: ReLU6 (0 to 6 clipping).
    4. Foldable BatchNorms: Merged into Conv weights during INT8 export.
    """
    def __init__(self, num_classes=28, width_mult=1.0, in_channels=3, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult

        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Base channel configuration
        c_stem = _make_divisible(24 * width_mult)
        c1 = _make_divisible(32 * width_mult)
        c2 = _make_divisible(48 * width_mult)
        c3 = _make_divisible(64 * width_mult)
        c4 = _make_divisible(96 * width_mult)
        c5 = _make_divisible(128 * width_mult)
        c6 = _make_divisible(160 * width_mult)

        # 1. Stem: Standard Conv 3x3 with stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU6(inplace=True)
        )

        # 2. Stage Blocks
        self.block1 = DepthwiseSeparableConv(c_stem, c1, stride=1)
        self.block2 = DepthwiseSeparableConv(c1, c2, stride=2)
        self.block3 = DepthwiseSeparableConv(c2, c3, stride=1)
        self.block4 = DepthwiseSeparableConv(c3, c4, stride=2)
        self.block5 = DepthwiseSeparableConv(c4, c5, stride=1)
        self.block6 = DepthwiseSeparableConv(c5, c6, stride=2)

        # 3. Global Pooling & Lightweight Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(c6, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_macs(self, input_size=(1, 3, 96, 96)):
        """Compute theoretical MACs for a given input tensor size."""
        total_macs = 0
        h, w = input_size[2], input_size[3]
        
        # Stem
        c_in = input_size[1]
        c_out = self.stem[0].out_channels
        h, w = h // 2, w // 2
        total_macs += c_in * 3 * 3 * c_out * h * w

        # Blocks
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]:
            if block.stride == 2:
                h, w = h // 2, w // 2
            in_c = block.dw_conv.in_channels
            out_c = block.pw_conv.out_channels
            # DW
            total_macs += in_c * 3 * 3 * h * w
            # PW
            total_macs += in_c * 1 * 1 * out_c * h * w

        # Head
        total_macs += self.classifier.in_features * self.classifier.out_features
        return total_macs

    def fold_batchnorms(self):
        """
        Folds all BatchNorm layers into preceding Conv layers for exact zero-overhead INT8 export.
        """
        folded_model = torch.quantization.fuse_modules(
            self,
            [
                ['stem.0', 'stem.1'],
                ['block1.dw_conv', 'block1.dw_bn'],
                ['block1.pw_conv', 'block1.pw_bn'],
                ['block2.dw_conv', 'block2.dw_bn'],
                ['block2.pw_conv', 'block2.pw_bn'],
                ['block3.dw_conv', 'block3.dw_bn'],
                ['block3.pw_conv', 'block3.pw_bn'],
                ['block4.dw_conv', 'block4.dw_bn'],
                ['block4.pw_conv', 'block4.pw_bn'],
                ['block5.dw_conv', 'block5.dw_bn'],
                ['block5.pw_conv', 'block5.pw_bn'],
                ['block6.dw_conv', 'block6.dw_bn'],
                ['block6.pw_conv', 'block6.pw_bn']
            ],
            inplace=False
        )
        return folded_model


def get_plantedge_model(num_classes=28, width_mult=1.0, in_channels=3):
    model = PlantEdgeNet(num_classes=num_classes, width_mult=width_mult, in_channels=in_channels)
    params = model.count_parameters()
    assert params < 100000, f"Error: Model has {params} params, exceeding 100K FPGA budget!"
    return model


if __name__ == "__main__":
    for w in [0.75, 1.0, 1.25]:
        m = get_plantedge_model(num_classes=28, width_mult=w)
        params = m.count_parameters()
        macs_96 = m.count_macs((1, 3, 96, 96))
        macs_64 = m.count_macs((1, 3, 64, 64))
        print(f"PlantEdgeNet (width={w:.2f}): {params:,} parameters | MACs @ 96px: {macs_96/1e6:.2f} M | MACs @ 64px: {macs_64/1e6:.2f} M")
