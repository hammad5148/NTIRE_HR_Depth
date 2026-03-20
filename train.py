import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

from misc import RunningAverageDict, compute_errors

# ──────────────────────────────────────────────────────────────────────────────
# Backend detection: native DAV2 (preferred) vs HuggingFace
# ──────────────────────────────────────────────────────────────────────────────
DAV2_NATIVE = False
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import (NormalizeImage, PrepareForNet,
                                                   Resize)
    from torchvision.transforms import Compose as TorchCompose
    DAV2_NATIVE = True
    print("[Backend] Native Depth Anything V2 available  ✓")
except ImportError:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print("[Backend] Native DAV2 not found; using HuggingFace relative-depth fallback.")


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss  –  Eigen et al., NeurIPS 2014.

    Removes global scale ambiguity; DAV2 official training uses variance_focus=0.85.
    Multiplied by 10 to bring it into the same numeric range as other losses.

        loss = sqrt( mean(d²) − λ·mean(d)² ) × 10
        d    = log(pred) − log(target)
    """
    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-3):
        super().__init__()
        self.vf  = variance_focus
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        pred   = torch.clamp(pred,   min=self.eps)
        target = torch.clamp(target, min=self.eps)

        p = pred[mask]   if mask is not None else pred.flatten()
        t = target[mask] if mask is not None else target.flatten()

        if p.numel() < 2:
            return pred.sum() * 0.0          # differentiable zero

        d    = torch.log(p) - torch.log(t)
        loss = torch.sqrt((d**2).mean() - self.vf * (d.mean()**2) + 1e-8) * 10.0
        return loss


class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient matching loss  –  MiDaS / DPT (Ranftl et al., 2020).

    Computes dx/dy gradient differences at 4 scales (step = 1,2,4,8 pixels).
    Forces sharp, correctly-positioned depth discontinuities at ToM edges,
    where pixel-wise losses fail because ToM surfaces reflect surroundings
    and create blurry depth predictions.
    """
    def __init__(self, scales: int = 4):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device)
        p = pred.unsqueeze(1).float()           # [B,1,H,W]
        t = target.unsqueeze(1).float()

        for s in range(self.scales):
            step = 2 ** s
            p_dx = p[:, :, :, step:] - p[:, :, :, :-step]
            t_dx = t[:, :, :, step:] - t[:, :, :, :-step]
            p_dy = p[:, :, step:, :] - p[:, :, :-step, :]
            t_dy = t[:, :, step:, :] - t[:, :, :-step, :]
            diff_x = (p_dx - t_dx).abs()
            diff_y = (p_dy - t_dy).abs()

            if mask is not None:
                m  = mask.unsqueeze(1).float()
                mx = m[:, :, :, step:] * m[:, :, :, :-step]
                my = m[:, :, step:, :] * m[:, :, :-step, :]
                total = total + (diff_x * mx).sum() / mx.sum().clamp(min=1)
                total = total + (diff_y * my).sum() / my.sum().clamp(min=1)
            else:
                total = total + diff_x.mean() + diff_y.mean()

        return total / self.scales


class CombinedDepthLoss(nn.Module):
    def __init__(self, w_silog=1.0, w_grad=0.8, w_l1=0.2, tom_weight=5.0):
        super().__init__()
        self.silog     = SILogLoss()
        self.grad      = MultiScaleGradientLoss(scales=4)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=0.1)
        self.w_silog   = w_silog
        self.w_grad    = w_grad
        self.w_l1      = w_l1
        self.tom_weight = tom_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor,
                tom_mask: torch.Tensor = None) -> torch.Tensor:
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        l_silog = self.silog(pred, target, valid_mask)
        l_grad  = self.grad(pred, target, valid_mask)

        l1_map  = self.smooth_l1(pred, target)          # [B,H,W]
        wmap    = valid_mask.float()
        if tom_mask is not None and tom_mask.any():
            wmap = wmap + tom_mask.float() * (self.tom_weight - 1.0)
        l_l1    = (l1_map * wmap).sum() / wmap.sum().clamp(min=1)

        return self.w_silog * l_silog + self.w_grad * l_grad + self.w_l1 * l_l1


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class BoosterDepthDataset(Dataset):

    def __init__(self, file_path: str, augment: bool = False,
                 input_size: int = 518, depth_scale: float = 1.0):
        self.augment     = augment
        self.input_size  = input_size
        self.depth_scale = depth_scale
        self.samples     = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.samples.append(parts)

        # Build image preprocessor
        if DAV2_NATIVE:
            self._tfm = TorchCompose([
                Resize(input_size, input_size,
                       keep_aspect_ratio=False,
                       ensure_multiple_of=14,
                       resize_method='lower_bound'),
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        else:
            self._hf = AutoImageProcessor.from_pretrained(
                "LiheYoung/depth-anything-large-hf")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _npy_to_rgb(path: str) -> np.ndarray:
        """Load .npy → H×W×3 uint8 RGB."""
        arr = np.load(path)
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr[..., :3]

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """H×W×3 uint8 → [3,H',W'] float32 tensor."""
        if DAV2_NATIVE:
            s = self._tfm({'image': img.astype(np.float32) / 255.0})
            return torch.from_numpy(s['image']).float()
        else:
            out = self._hf(images=img, return_tensors="pt")
            return out['pixel_values'].squeeze(0).float()

    def __getitem__(self, idx: int):
        parts = self.samples[idx]
        try:
            raw = self._npy_to_rgb(parts[0])
            tgt = np.load(parts[1]).astype(np.float32) * self.depth_scale
            msk = np.load(parts[2]).astype(bool)
            tom = np.load(parts[3]).astype(bool) if len(parts) >= 4 else None
        except Exception as e:
            print(f"[Dataset] Load error idx={idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            import cv2
            # 1. Random crop + scale
            if random.random() < 0.3:
                h, w = raw.shape[:2]
                scale = random.uniform(0.8, 1.0)
                nh, nw = int(h*scale), int(w*scale)
                y0 = random.randint(0, h-nh)
                x0 = random.randint(0, w-nw)
                raw = cv2.resize(raw[y0:y0+nh, x0:x0+nw], (w,h), interpolation=cv2.INTER_LANCZOS4)
                tgt = cv2.resize(tgt[y0:y0+nh, x0:x0+nw], (w,h), interpolation=cv2.INTER_NEAREST)
                msk = cv2.resize(msk[y0:y0+nh, x0:x0+nw].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)
                if tom is not None:
                    tom = cv2.resize(tom[y0:y0+nh, x0:x0+nw].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)

            # 2. Horizontal flip
            if random.random() < 0.5:
                raw = np.ascontiguousarray(raw[:, ::-1, :])
                tgt = np.ascontiguousarray(tgt[:, ::-1])
                msk = np.ascontiguousarray(msk[:, ::-1])
                if tom is not None:
                    tom = np.ascontiguousarray(tom[:, ::-1])

            # 3. Color jitter
            img_t = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.0
            bri = 1.0 + random.uniform(-0.20, 0.20)
            con = 1.0 + random.uniform(-0.20, 0.20)
            sat = 1.0 + random.uniform(-0.20, 0.20)
            img_t = TF.adjust_brightness(img_t, bri)
            img_t = TF.adjust_contrast(img_t,   con)
            img_t = TF.adjust_saturation(img_t, sat)
            raw   = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # 4. Gaussian noise
            if random.random() < 0.2:
                noise = np.random.normal(0, 5, raw.shape).astype(np.int16)
                raw = np.clip(raw.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        pv   = self._preprocess(raw)
        tgt_t = torch.from_numpy(tgt).float()
        msk_t = torch.from_numpy(msk).bool()
        tom_t = torch.from_numpy(tom).bool() if tom is not None \
                else torch.zeros_like(msk_t)

        return pv, tgt_t, msk_t, tom_t


def _pad_to_max(tensors: list) -> torch.Tensor:
    """Pad 2-D (or 3-D) tensors to the same H×W, then stack."""
    mh = max(t.shape[-2] for t in tensors)
    mw = max(t.shape[-1] for t in tensors)
    padded = [F.pad(t, (0, mw - t.shape[-1], 0, mh - t.shape[-2]))
              for t in tensors]
    return torch.stack(padded)


def collate_fn(batch):
    pv, tgts, masks, toms = zip(*batch)
    return (_pad_to_max(list(pv)),
            _pad_to_max(list(tgts)),
            _pad_to_max(list(masks)).bool(),
            _pad_to_max(list(toms)).bool())


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lse_align(pred: np.ndarray, target: np.ndarray,
              mask: np.ndarray = None) -> np.ndarray:
    """
    Least-Squares scale+shift alignment: pred' = s·pred + t
    Official NTIRE mono-track evaluation protocol (following MiDaS).
    """
    p = pred[mask]   if mask is not None else pred.flatten()
    t = target[mask] if mask is not None else target.flatten()
    A          = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_    = np.linalg.lstsq(A, t, rcond=None)
    s, shift   = max(float(sol[0]), 1e-6), float(sol[1])
    return np.clip(s * pred + shift, 1e-3, None)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_DAV2_CFG = {
    'vits': dict(encoder='vits', features=64,  out_channels=[48,  96,  192,  384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96,  192, 384,  768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


def build_model(args) -> nn.Module:
    if DAV2_NATIVE:
        model = DepthAnythingV2(**_DAV2_CFG[args.encoder], max_depth=args.max_depth)

        # Load metric weights: user-supplied path first, then HuggingFace auto-download
        if args.pretrained_weights and os.path.exists(args.pretrained_weights):
            sd = torch.load(args.pretrained_weights, map_location='cpu',
                            weights_only=True)
            model.load_state_dict(sd, strict=True)
            print(f"[Model] Loaded metric weights from {args.pretrained_weights}")
        else:
            try:
                from huggingface_hub import hf_hub_download
                enc  = args.encoder
                fname = f"depth_anything_v2_metric_hypersim_{enc}.pth"
                path  = hf_hub_download(
                    repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
                    filename=fname,
                    local_dir=os.path.join(args.checkpoints_dir, "pretrained"))
                sd = torch.load(path, map_location='cpu', weights_only=True)
                model.load_state_dict(sd, strict=True)
                print(f"[Model] Auto-downloaded metric weights ({enc}) from HF")
            except Exception as e:
                print(f"[Model] Warning – metric weights not loaded: {e}")
                print("[Model] Using random metric head + pre-trained ViT encoder.")
    else:
        model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-large-hf")
        print("[Model] HuggingFace relative-depth model loaded.")

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        try:
            raw = torch.load(args.load_checkpoint, map_location='cpu',
                             weights_only=False)
            sd  = raw.state_dict() if hasattr(raw, 'state_dict') else raw
            model.load_state_dict(sd, strict=False)
            print(f"[Model] Resumed from {args.load_checkpoint}")
        except Exception as e:
            print(f"[Model] Checkpoint load failed: {e}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, args, model, dl_train, dl_val, device):
        self.args     = args
        self.model    = model
        self.device   = device
        self.dl_train = dl_train
        self.dl_val   = dl_val

        self.criterion = CombinedDepthLoss(
            w_silog=1.0, w_grad=0.8, w_l1=0.2, tom_weight=5.0)

        # ── Differential LR: lower for ViT backbone, higher for metric head ──
        raw = unwrap(model)
        if DAV2_NATIVE and hasattr(raw, 'pretrained'):
            bb_ids = {id(p) for p in raw.pretrained.parameters()}
            bb     = [p for p in raw.parameters() if id(p) in bb_ids]
            hd     = [p for p in raw.parameters() if id(p) not in bb_ids]
            groups = [{'params': bb, 'lr': args.lr_backbone, 'name': 'backbone'},
                      {'params': hd, 'lr': args.lr_head,     'name': 'head'}]
        else:
            groups = [{'params': raw.parameters(), 'lr': args.lr_head, 'name': 'all'}]

        self.opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

        # Cosine schedule (primary) + ReduceLROnPlateau (safety net)
        self.sched_cos  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=max(args.epochs // 2, 1), T_mult=1, eta_min=1e-7)
        self.sched_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=2, min_lr=1e-7)

        self.scaler = GradScaler(enabled=args.amp)

        self.best_abs_rel  = float('inf')
        self.no_improve    = 0
        self.total_metrics = RunningAverageDict()
        self.is_main       = (not args.use_ddp) or (dist.get_rank() == 0)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _dev(self, *tensors):
        return [t.to(self.device, non_blocking=True) for t in tensors]

    def _forward(self, pv: torch.Tensor, hw: tuple) -> torch.Tensor:
        if DAV2_NATIVE:
            depth = self.model(pv)
        else:
            depth = self.model(pixel_values=pv).predicted_depth
        depth = F.interpolate(
            depth.unsqueeze(1), size=hw, mode='bilinear', align_corners=False
        ).squeeze(1)
        return torch.clamp(depth, min=1e-3)

    # ── training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()

        # Epoch 0-1: freeze ViT backbone to not destroy strong synthetic priors
        # Epoch 2+: unfreeze at low LR to allow domain adaptation
        if DAV2_NATIVE and hasattr(unwrap(self.model), 'pretrained'):
            freeze = (epoch < 2)
            for p in unwrap(self.model).pretrained.parameters():
                p.requires_grad_(not freeze)
            if self.is_main:
                print(f"[Train] Backbone {'FROZEN' if freeze else 'unfrozen'} "
                      f"(epoch {epoch})")

        if self.args.use_ddp and hasattr(self.dl_train.sampler, 'set_epoch'):
            self.dl_train.sampler.set_epoch(epoch)

        ep_loss = 0.0
        pbar    = tqdm(self.dl_train, desc=f"Ep {epoch:02d} | loss=-.----",
                       disable=not self.is_main, leave=True)

        for pv, tgt, vmsk, tmsk in pbar:
            pv, tgt, vmsk, tmsk = self._dev(pv, tgt, vmsk, tmsk)
            self.opt.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.amp):
                depth = self._forward(pv, tgt.shape[-2:])
                loss  = self.criterion(depth, tgt, vmsk,
                                       tmsk if tmsk.any() else None)

            if not torch.isfinite(loss):
                print("[Train] Non-finite loss, skipping batch.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()

            ep_loss += loss.item()
            if self.is_main:
                pbar.set_description(f"Ep {epoch:02d} | loss={loss.item():.4f}")

        self.sched_cos.step(epoch)

        if self.is_main:
            lr_b = self.opt.param_groups[0]['lr']
            lr_h = self.opt.param_groups[-1]['lr']
            avg  = ep_loss / max(len(self.dl_train), 1)
            print(f"[Train] Ep {epoch:02d}  avg_loss={avg:.5f}  "
                  f"lr_bb={lr_b:.2e}  lr_hd={lr_h:.2e}")

    # ── validation epoch ──────────────────────────────────────────────────────

    def validate(self, epoch: int) -> float:
        self.model.eval()
        metrics = RunningAverageDict()

        with torch.no_grad():
            for pv, tgt, vmsk, tmsk in tqdm(
                    self.dl_val, desc=f"Val  {epoch:02d}",
                    disable=not self.is_main, leave=False):
                pv, tgt, vmsk, tmsk = self._dev(pv, tgt, vmsk, tmsk)

                with autocast(enabled=self.args.amp):
                    depth = self._forward(pv, tgt.shape[-2:])

                # Per-image LSE scale+shift alignment (NTIRE mono protocol)
                for b in range(depth.shape[0]):
                    vm = vmsk[b].cpu().numpy().astype(bool)
                    if vm.sum() < 10:
                        continue

                    p_np = depth[b].float().cpu().numpy()
                    t_np = tgt[b].float().cpu().numpy()

                    p_aligned = lse_align(p_np, t_np, vm)

                    p_v = p_aligned[vm]
                    t_v = t_np[vm]
                    pos = (p_v > 0) & (t_v > 0)
                    if pos.sum() < 5:
                        continue

                    met = compute_errors(t_v[pos], p_v[pos])
                    metrics.update(met)
                    self.total_metrics.update(met)

        abs_rel = float('inf')
        if self.is_main:
            md      = {k: round(v, 5) for k, v in metrics.get_value().items()}
            abs_rel = md.get('abs_rel', float('inf'))
            print(f"[Val]  Ep {epoch:02d}  {md}")

            os.makedirs(self.args.checkpoints_dir, exist_ok=True)
            torch.save(unwrap(self.model).state_dict(),
                       os.path.join(self.args.checkpoints_dir, 'model_latest.pt'))

            if abs_rel < self.best_abs_rel:
                self.best_abs_rel = abs_rel
                self.no_improve   = 0
                torch.save(unwrap(self.model).state_dict(),
                           os.path.join(self.args.checkpoints_dir, 'model_best.pt'))
                print(f"[Val]  ✓ Best model  abs_rel={abs_rel:.5f}  → model_best.pt")
            else:
                self.no_improve += 1

            if (epoch + 1) % self.args.save_every == 0:
                torch.save(unwrap(self.model).state_dict(),
                           os.path.join(self.args.checkpoints_dir,
                                        f'model_ep{epoch:02d}.pt'))

        self.sched_plat.step(abs_rel if abs_rel < float('inf') else 1.0)
        return abs_rel

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            if self.no_improve >= self.args.patience:
                if self.is_main:
                    print(f"[Train] Early stop after {self.no_improve} epochs "
                          f"without improvement.")
                break

        if self.is_main:
            final = {k: round(v, 5)
                     for k, v in self.total_metrics.get_value().items()}
            print(f"\n[Done] Average metrics across all epochs: {final}")
            print(f"[Done] Best abs_rel: {self.best_abs_rel:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("NTIRE 2026 HR Depth – ToM Surfaces")

    p.add_argument('--train_txt', default=
        "/kaggle/working/NTIRE_HR_Depth/dataset_paths/train_extended.txt")
    p.add_argument('--checkpoints_dir', default=
        "/kaggle/working/NTIRE_HR_Depth/checkpoints_new")
    p.add_argument('--pretrained_weights', default="/kaggle/working/NTIRE_HR_Depth/checkpoints_new/pretrained/depth_anything_v2_metric_hypersim_vitl.pth",
        help="Path to DAV2 metric .pth  (depth_anything_v2_metric_hypersim_vitl.pth)")
    p.add_argument('--load_checkpoint', default=None,
        help="Resume fine-tuning from this checkpoint")

    p.add_argument('--encoder',      default='vitl', choices=['vits','vitb','vitl'])
    p.add_argument('--max_depth',    type=float, default=20.0,
        help="Max depth in metres for metric head (20m for Booster indoor)")
    p.add_argument('--depth_scale',  type=float, default=1.0,
        help="Multiply raw depth values by this; use 0.001 if stored in mm")

    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--epochs',       type=int,   default=25)
    p.add_argument('--lr_backbone',  type=float, default=5e-6)
    p.add_argument('--lr_head',      type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=5e-3)
    p.add_argument('--input_size',   type=int,   default=518,
        help="Image size fed to DAV2 (must be divisible by 14)")
    p.add_argument('--amp',          action='store_true', default=True)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--save_every',   type=int,   default=2)

    p.add_argument('--gpu_ids',    default='0,1')
    p.add_argument('--use_ddp',    action='store_true')
    p.add_argument('--local_rank', type=int, default=0)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    args = parse_args()
    set_seed(42)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device(f"cuda:{local_rank}")
        print(f"[DDP] rank={dist.get_rank()}  device={DEVICE}")
    else:
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')
                   if g.strip().isdigit() and torch.cuda.is_available()]
        DEVICE  = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        print(f"[DP]  GPUs={gpu_ids or 'CPU'}  primary={DEVICE}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(DEVICE)

    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    elif len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[DP]  Replicated across {gpu_ids}.")

    total = sum(p.numel() for p in unwrap(model).parameters())
    train_p = sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)
    print(f"[Model] {total/1e6:.1f}M total params  |  {train_p/1e6:.1f}M trainable")

    # ── Dataset (two instances: different augment flags) ───────────────────────
    txt = os.path.abspath(args.train_txt)
    print(f"[Data] {txt}")

    _base  = BoosterDepthDataset(txt, augment=False,
                                  input_size=args.input_size,
                                  depth_scale=args.depth_scale)
    n      = len(_base)
    rng    = torch.Generator().manual_seed(42)
    idx    = torch.randperm(n, generator=rng).tolist()
    val_n  = int(0.2 * n)
    v_idx, tr_idx = idx[:val_n], idx[val_n:]

    ds_tr = Subset(BoosterDepthDataset(txt, augment=True,
                                        input_size=args.input_size,
                                        depth_scale=args.depth_scale), tr_idx)
    ds_vl = Subset(BoosterDepthDataset(txt, augment=False,
                                        input_size=args.input_size,
                                        depth_scale=args.depth_scale), v_idx)

    print(f"[Data] Train={len(ds_tr)}  Val={len(ds_vl)}")

    kw = dict(collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    if args.use_ddp:
        dl_tr = DataLoader(ds_tr, args.batch_size,
                           sampler=DistributedSampler(ds_tr, shuffle=True),
                           num_workers=4, **kw)
        dl_vl = DataLoader(ds_vl, args.batch_size,
                           sampler=DistributedSampler(ds_vl, shuffle=False),
                           num_workers=2, **kw)
    else:
        dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True,
                           num_workers=4, **kw)
        dl_vl = DataLoader(ds_vl, args.batch_size, shuffle=False,
                           num_workers=2, **kw)

    # ── Train ──────────────────────────────────────────────────────────────────
    try:
        Trainer(args, model, dl_tr, dl_vl, DEVICE).train()
    except Exception as exc:
        import traceback
        print(f"[Error] {exc}")
        traceback.print_exc()
    finally:
        if args.use_ddp:
            dist.destroy_process_group()
