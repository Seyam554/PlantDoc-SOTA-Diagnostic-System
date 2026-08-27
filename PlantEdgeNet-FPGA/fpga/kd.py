r"""
Shared knowledge-distillation utilities: multi-teacher soft targets,
DIST relational loss, and attention-transfer feature loss.

Used by fpga/train_fpga.py and fpga/distill.py.

Teachers are loaded from repo checkpoints produced by train_sota.py / train.py /
fpga/train_teacher.py (dict with "state_dict" + "model_name"). Architectures:
  * dinov2_* / convnext* / swin*  -> models_sota.get_sota_model   (needs timm for convnext/swin)
  * vgg16 / resnet50 / mobilenet_v2 / inception_v3 -> models.get_model
  * a timm name (e.g. "efficientnet_b3") -> timm.create_model      (needs timm)

Each Teacher exposes .logits(x) and, when possible, .features(x) (B,C,h,w) for
attention transfer. DINOv2 (token output) has no spatial map -> features() = None.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Teacher:
    def __init__(self, model, in_size, name, spatial=True):
        self.model = model.eval()
        self.in_size = int(in_size)
        self.name = name
        self.spatial = spatial
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def _resize(self, x):
        if x.shape[-1] != self.in_size:
            x = F.interpolate(x.float(), size=self.in_size, mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def logits(self, x):
        out = self.model(self._resize(x))
        return out[0] if isinstance(out, (tuple, list)) else out

    @torch.no_grad()
    def features(self, x):
        if not self.spatial:
            return None
        m = self.model
        xr = self._resize(x)
        try:
            if hasattr(m, "forward_features"):
                f = m.forward_features(xr)
            elif hasattr(m, "model") and hasattr(m.model, "forward_features"):
                f = m.model.forward_features(xr)
            else:
                return None
        except Exception:
            return None
        if isinstance(f, (tuple, list)):
            f = f[-1]
        if f.dim() == 3:            # tokens (B, N, C) -> treat as non-spatial
            return None
        if f.dim() != 4:
            return None
        return f


def load_teachers(paths, num_classes, device, verbose=True):
    teachers = []
    for p in paths or []:
        try:
            if os.path.exists(p):
                ck = torch.load(p, map_location="cpu")
                name = ck.get("model_name", os.path.splitext(os.path.basename(p))[0])
                in_size = int(ck.get("args", {}).get("img_size", 224))
                model, spatial = _build(name, num_classes)
                model.load_state_dict(ck["state_dict"], strict=False)
            else:
                # bare timm name: ImageNet backbone + RANDOM classifier head.
                # Its LOGITS are meaningless for PlantDoc -> use only for
                # attention-transfer features. Fine-tune it first with
                # fpga/train_teacher.py for a real logit teacher.
                import timm
                name = p
                model = timm.create_model(p, pretrained=True, num_classes=num_classes)
                in_size, spatial = 224, True
                print(f"[kd][WARN] '{p}' has an untrained head - its soft labels will be noise. "
                      f"Run: python fpga/train_teacher.py --arch {p} --data-dir <cropped> "
                      f"then pass that .pth instead.")
            teachers.append(Teacher(model, in_size, name, spatial).to(device))
            if verbose:
                print(f"[kd] teacher: {name}  in={in_size}px  spatial={spatial}")
        except Exception as e:
            print(f"[kd][warn] could not load teacher '{p}': {e}")
    return teachers


def _build(name, num_classes):
    nl = name.lower()
    if any(k in nl for k in ("dinov2", "convnext", "swin")):
        from models_sota import get_sota_model
        return get_sota_model(arch=name, num_classes=num_classes), ("dinov2" not in nl)
    if nl in ("vgg16", "resnet50", "mobilenet_v2", "inception_v3"):
        from models import get_model
        return get_model(nl, num_classes=num_classes, pretrained=False), True
    import timm
    return timm.create_model(name, pretrained=False, num_classes=num_classes), True


@torch.no_grad()
def teacher_soft(teachers, x, T):
    """Averaged softened teacher distribution (B,K). None if no teachers."""
    if not teachers:
        return None
    acc = None
    for t in teachers:
        p = F.softmax(t.logits(x) / T, dim=1)
        acc = p if acc is None else acc + p
    return acc / len(teachers)


def kd_kl(student_logits, teacher_probs, T):
    return F.kl_div(F.log_softmax(student_logits / T, 1), teacher_probs,
                    reduction="batchmean") * (T * T)


def _pearson(a, b, dim):
    a = a - a.mean(dim, keepdim=True)
    b = b - b.mean(dim, keepdim=True)
    num = (a * b).sum(dim)
    den = a.norm(dim=dim) * b.norm(dim=dim) + 1e-8
    return num / den


def dist_loss(student_logits, teacher_probs, T=1.0, beta=1.0, gamma=1.0):
    """DIST (Huang et al. 2022): match inter-class and intra-class relations
    instead of exact probabilities - robust when teacher >> student."""
    ps = F.softmax(student_logits / T, dim=1)
    pt = teacher_probs
    inter = 1 - _pearson(ps, pt, dim=1).mean()          # per-sample, across classes
    intra = 1 - _pearson(ps, pt, dim=0).mean()          # per-class, across batch
    return beta * inter + gamma * intra


def at_loss(student_feat, teachers, x):
    """Attention transfer (Zagoruyko & Komodakis): match spatial attention maps
    (mean of squared channel responses, L2-normalized) student<->teacher."""
    if student_feat is None:
        return 0.0

    def amap(f):
        a = f.pow(2).mean(1).flatten(1)
        return F.normalize(a, dim=1)

    sa = amap(student_feat)
    losses = []
    for t in teachers:
        tf = t.features(x)
        if tf is None:
            continue
        tf = F.interpolate(tf, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(sa, amap(tf)))
    if not losses:
        return student_feat.new_zeros(())
    return torch.stack(losses).mean()
