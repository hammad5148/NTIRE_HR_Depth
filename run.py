import argparse
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.transforms import functional as TF
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Backend detection  (mirrors train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
DAV2_NATIVE = False
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import (NormalizeImage, PrepareForNet,
                                                   Resize)
    from torchvision.transforms import Compose as TorchCompose
    DAV2_NATIVE = True
    print("[Backend] Native Depth Anything V2  ✓")
except ImportError:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print("[Backend] HuggingFace fallback (install DAV2 for best results).")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_DEPTH_CM   = 1000.0   # competition range [0, 1000] cm
MAX_DEPTH_M    = MAX_DEPTH_CM / 100.0   # 10 m (== 1000 cm)

_DAV2_CFG = {
    'vits': dict(encoder='vits', features=64,  out_channels=[48,  96,  192,  384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96,  192, 384,  768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSOR  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    def __init__(self, input_size: int = 518):
        self.input_size = input_size
        if DAV2_NATIVE:
            self._tfm = TorchCompose([
                Resize(
                    input_size, input_size,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        else:
            self._hf = AutoImageProcessor.from_pretrained(
                "LiheYoung/depth-anything-large-hf")

    def __call__(self, img_uint8: np.ndarray) -> torch.Tensor:
        """Returns [3, H', W'] float32 tensor (no batch dim)."""
        if DAV2_NATIVE:
            s = self._tfm({'image': img_uint8.astype(np.float32) / 255.0})
            return torch.from_numpy(s['image']).float()
        else:
            out = self._hf(images=img_uint8, return_tensors="pt")
            return out['pixel_values'].squeeze(0).float()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

class HFWrapper(nn.Module):
    """Thin wrapper so HuggingFace model returns a bare tensor, not a dataclass."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).predicted_depth


def _remap_state_dict(sd: dict) -> dict:
    def strip(sd, prefix):
        if any(k.startswith(prefix) for k in sd):
            return {k[len(prefix):] if k.startswith(prefix) else k: v
                    for k, v in sd.items()}
        return sd

    sd = strip(sd, 'module.')
    sd = strip(sd, 'model.')
    return sd


def load_model(ckpt_path: str, encoder: str, max_depth: float,
               device: torch.device) -> nn.Module:

    # ── Build architecture ────────────────────────────────────────────────────
    if DAV2_NATIVE:
        model = DepthAnythingV2(**_DAV2_CFG[encoder], max_depth=max_depth)
    else:
        base  = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-large-hf")
        model = HFWrapper(base)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd  = raw.state_dict() if hasattr(raw, 'state_dict') else raw
    sd  = _remap_state_dict(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [Ckpt] Missing  keys: {len(missing):3d}  "
              f"(first few: {[k for k in missing[:3]]})")
    if unexpected:
        print(f"  [Ckpt] Unexpect keys: {len(unexpected):3d}  "
              f"(first few: {[k for k in unexpected[:3]]})")
    if not missing and not unexpected:
        print("  [Ckpt] Loaded perfectly (strict=True equivalent).")

    total = sum(p.numel() for p in model.parameters())
    print(f"  [Ckpt] {total/1e6:.1f}M params  |  checkpoint: {ckpt_path}")

    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
#  LSE ALIGNMENT  (NTIRE official protocol — mirrors train.py)
# ══════════════════════════════════════════════════════════════════════════════

def lse_align(pred: np.ndarray, target: np.ndarray,
              mask: np.ndarray = None) -> np.ndarray:

    p = pred[mask].flatten()   if mask is not None else pred.flatten()
    t = target[mask].flatten() if mask is not None else target.flatten()

    if len(p) < 2:
        return pred

    A       = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, t, rcond=None)
    s, sh   = max(float(sol[0]), 1e-6), float(sol[1])
    return np.clip(s * pred + sh, 1e-3, None)


# ══════════════════════════════════════════════════════════════════════════════
#  DEPTH → CM CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def depth_to_cm(depth_raw: np.ndarray,
                max_depth_m: float,
                scene_max_cm: float,
                invert: bool = False) -> np.ndarray:
    if DAV2_NATIVE and not invert:
        # Native metric DAV2: metres -> cm directly, close=small (correct)
        depth_cm = depth_raw * 100.0
        depth_cm = np.clip(depth_cm, 0.0, scene_max_cm)

    else:
        # HF disparity OR forced inversion
        # Step 1: percentile-clip to suppress outlier pixels at edges
        p1  = float(np.percentile(depth_raw, 1))
        p99 = float(np.percentile(depth_raw, 99))
        if p99 - p1 < 1e-8:
            return np.zeros_like(depth_raw, dtype=np.float32)
        depth_clipped = np.clip(depth_raw, p1, p99)

        # Step 2: normalise to [0,1]  ->  HF: close=1.0, far=0.0
        depth_norm = (depth_clipped - p1) / (p99 - p1)

        # Step 3: INVERT  ->  close=0.0, far=1.0  (matches metric convention)
        depth_norm = 1.0 - depth_norm

        # Step 4: scale to [0, scene_max_cm]
        # scene_max_cm is the realistic max depth of the scene in cm.
        # Booster tabletop ~ 200 cm. Too high -> too bright; too low -> too dark.
        depth_cm = depth_norm * scene_max_cm

    return np.clip(depth_cm, 0.0, MAX_DEPTH_CM).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE  (with TTA)
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:

    def __init__(self, model: nn.Module, preprocessor: Preprocessor,
                 device: torch.device, use_tta: bool = True,
                 amp: bool = True):
        self.model  = model
        self.prep   = preprocessor
        self.device = device
        self.tta    = use_tta
        self.amp    = amp and (device.type == 'cuda')

    @torch.no_grad()
    def _pass(self, img_uint8: np.ndarray, out_hw: tuple) -> np.ndarray:
        """Single forward pass: img → model → bilinear upsample to out_hw."""
        pv = self.prep(img_uint8).unsqueeze(0).to(self.device)   # [1,3,H',W']

        with autocast(enabled=self.amp):
            depth = self.model(pv)                               # [1, H', W']  or  [H', W']

        # Ensure 4-D for interpolate
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(
            depth.float(), size=out_hw,
            mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()                                 # [H, W]

        return np.clip(depth, 1e-3, None)

    @torch.no_grad()
    def predict(self, img_uint8: np.ndarray) -> np.ndarray:

        orig_hw = img_uint8.shape[:2]   # (H, W)
        preds   = []

        # ── Pass 1: original ──────────────────────────────────────────────────
        preds.append(self._pass(img_uint8, orig_hw))

        if self.tta:
            # ── Pass 2: horizontal flip ───────────────────────────────────────
            img_flip  = np.ascontiguousarray(img_uint8[:, ::-1, :])
            d_flip    = self._pass(img_flip, orig_hw)
            preds.append(d_flip[:, ::-1])                        # flip depth back

            # ── Pass 3: 1.2× upscale (richer context) ────────────────────────
            H, W      = orig_hw
            H2, W2    = int(H * 1.2), int(W * 1.2)
            img_large = cv2.resize(img_uint8, (W2, H2),
                                   interpolation=cv2.INTER_LANCZOS4)
            d_large   = self._pass(img_large, (H2, W2))
            d_large_r = cv2.resize(d_large, (W, H),             # back to original
                                   interpolation=cv2.INTER_LINEAR)
            preds.append(d_large_r)

        # ── Ensemble in log-depth space ───────────────────────────────────────
        log_avg = np.mean([np.log(np.clip(p, 1e-6, None)) for p in preds], axis=0)
        return np.exp(log_avg).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """Load PNG / JPG / NPY → H×W×3 uint8 RGB numpy array."""
    if path.lower().endswith('.npy'):
        arr = np.load(path)
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr[..., :3]
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2 cannot open: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_png(depth_cm: np.ndarray, output_path: str) -> None:

    depth = depth_cm.astype(np.float32)
    depth = np.clip(depth, 0.0, 1000.0)
    depth_16bit = (depth / 1000.0 * 65535.0).astype(np.uint16)
    img = Image.fromarray(depth_16bit, mode="I;16")
    img.save(output_path)


def save_dir_for(filename: str, outdir: str) -> str:

    parts = filename.replace('\\', '/').split('/')
    if len(parts) >= 3:
        return os.path.join(outdir, parts[-3])
    elif len(parts) >= 2:
        return os.path.join(outdir, parts[-2])
    return outdir


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="NTIRE 2026 HR Depth – Inference & Submission Script")

    # ── Input / output ────────────────────────────────────────────────────────
    p.add_argument('--img-path', type=str,
        default="./dataset_paths/test.txt",
        help="Text file of image paths, a glob, or a directory.")
    p.add_argument('--outdir', type=str,
        default='./vis_depth',
        help="Root output directory for submission PNGs.")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    p.add_argument('-p', '--checkpoints', type=str,
        default="./checkpoints_new/47.37_weights.pt",
        help="Path to fine-tuned .pt state-dict checkpoint.")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl'],
        help="DAV2 encoder size (must match training). Default: vitl (335M).")
    p.add_argument('--max_depth', type=float, default=20.0,
        help="max_depth used during training (metres). "
             "Must match train.py (default 20.0 for Booster indoor).")
    p.add_argument('--scene_max_cm', type=float, default=200.0,
        help="Realistic max scene depth in cm for HF output rescaling. "
             "Booster tabletop ~ 150-250 cm. "
             "Increase if PNG looks too dark; decrease if too bright. "
             "(default 200). Has no effect when using native metric DAV2.)")
    p.add_argument('--invert', action='store_true',
        help="Force depth inversion. Auto-applied for HF backend. "
             "Use this flag if native DAV2 output also looks inverted.")
    p.add_argument('--input_size', type=int, default=518,
        help="Spatial size fed to DAV2 (must be divisible by 14).")

    # ── Output size ───────────────────────────────────────────────────────────
    p.add_argument('-width',  type=int, default=None,
        help="Force output width.  Default: use original image width.")
    p.add_argument('-height', type=int, default=None,
        help="Force output height. Default: use original image height.")

    # ── TTA / speed ───────────────────────────────────────────────────────────
    p.add_argument('--no_tta', action='store_true',
        help="Disable Test-Time Augmentation (3× faster, slightly lower accuracy).")
    p.add_argument('--no_amp', action='store_true',
        help="Disable AMP float16 (use if you get NaN on old GPUs).")

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_TTA = not args.no_tta
    USE_AMP = not args.no_amp

    print("=" * 60)
    print("  NTIRE 2026 HR Depth  —  Inference")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Backend     : {'Native DAV2' if DAV2_NATIVE else 'HuggingFace (disparity)'}")
    print(f"  Encoder     : {args.encoder}")
    print(f"  max_depth   : {args.max_depth} m")
    print(f"  TTA         : {'ON  (H-flip + 1.2× scale)' if USE_TTA else 'OFF'}")
    print(f"  AMP         : {'ON (float16)' if USE_AMP else 'OFF'}")
    auto_invert = (not DAV2_NATIVE) or args.invert
    print(f"  scene_max_cm: {args.scene_max_cm:.0f} cm  (tune if output too bright/dark)")
    print(f"  Invert      : {'YES (HF disparity → metric depth)' if auto_invert else 'NO (native metric DAV2)'}")
    print(f"  Encoding    : uint16 = depth_cm / 1000 × 65535  (official spec)")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(
        ckpt_path  = os.path.abspath(args.checkpoints),
        encoder    = args.encoder,
        max_depth  = args.max_depth,
        device     = DEVICE,
    )

    prep   = Preprocessor(input_size=args.input_size)
    engine = InferenceEngine(model, prep, DEVICE,
                             use_tta=USE_TTA, amp=USE_AMP)

    # ── Collect file list ─────────────────────────────────────────────────────
    img_path = os.path.abspath(args.img_path)

    if os.path.isfile(img_path) and img_path.endswith('.txt'):
        # Text file: each line = image path (only first column used)
        with open(img_path) as f:
            filenames = [l.strip().split()[0] for l in f
                         if l.strip() and not l.startswith('#')]
    elif '*' in img_path:
        filenames = sorted(glob(img_path))
    elif os.path.isdir(img_path):
        filenames = sorted([
            os.path.join(img_path, fn)
            for fn in os.listdir(img_path)
            if not fn.startswith('.')
            and fn.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))
        ])
    else:
        filenames = [img_path]   # single file

    print(f"\n  Images found: {len(filenames)}")
    os.makedirs(args.outdir, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    auto_invert = (not DAV2_NATIVE) or args.invert   # HF always inverted

    for filename in tqdm(filenames, desc="Inferring", unit="img"):
        if not os.path.exists(filename):
            print(f"[Warn] Not found, skipping: {filename}")
            continue

        # 1. Load image as uint8 RGB
        img_rgb = load_image(filename)

        # 2. Determine output (H, W) — use forced size or original size
        orig_h, orig_w = img_rgb.shape[:2]
        out_h = args.height if args.height else orig_h
        out_w = args.width  if args.width  else orig_w

        # 3. Resize to competition output resolution if different from original
        if (out_h, out_w) != (orig_h, orig_w):
            img_rgb = cv2.resize(img_rgb, (out_w, out_h),
                                 interpolation=cv2.INTER_LANCZOS4)

        # 4. Run inference (+ TTA if enabled)
        depth_raw = engine.predict(img_rgb)       # float32 H×W, raw model units

        # ── Sanity diagnostic on first image ──────────────────────────────────
        if filename == filenames[0]:
            print(f"\n[Diag] Raw model output  — "
                  f"min={depth_raw.min():.4f}  max={depth_raw.max():.4f}  "
                  f"mean={depth_raw.mean():.4f}")
            print(f"[Diag] Inversion applied : {auto_invert}")
            print(f"[Diag] scene_max_cm      : {args.scene_max_cm:.0f} cm")

        # 5. Convert raw depth → cm with correct inversion + scene scale
        depth_cm = depth_to_cm(depth_raw,
                                max_depth_m=args.max_depth,
                                scene_max_cm=args.scene_max_cm,
                                invert=auto_invert)

        if filename == filenames[0]:
            u16 = (depth_cm / 1000.0 * 65535.0).astype('uint16')
            print(f"[Diag] After conversion  — "
                  f"min={depth_cm.min():.1f} cm  max={depth_cm.max():.1f} cm  "
                  f"mean={depth_cm.mean():.1f} cm")
            print(f"[Diag] uint16 stats      — "
                  f"min={u16.min()}  max={u16.max()}  mean={u16.mean():.0f}")
            print(f"[Diag] GUIDE: mean uint16 ~ 5000-15000 matches Booster GT (dark PNG)")
            print(f"[Diag]   Too bright (mean>20000): lower --scene_max_cm (try 100-150)")
            print(f"[Diag]   Too dark   (mean< 2000): raise --scene_max_cm (try 300-400)\n")

        # ── Mirror folder structure for competition submission ────────────────
        save_dir = save_dir_for(filename, args.outdir)
        os.makedirs(save_dir, exist_ok=True)
        stem = Path(filename).stem

        # 6. Save directly as uint16 PNG — official organiser snippet:
        #      depth = np.clip(depth_cm, 0.0, 1000.0)
        #      depth_16bit = (depth / 1000.0 * 65535.0).astype(np.uint16)
        #      img = Image.fromarray(depth_16bit, mode="I;16")
        png_path = os.path.join(save_dir, f"{stem}.png")
        save_png(depth_cm, png_path)

    print(f"\n\u2713 Done.  {len(filenames)} images saved to: {args.outdir}")
    print(f"   .png  (uint16 I;16) : <outdir>/<scene>/<stem>.png  \u2190 submit this")
    print("\nVerify a single output (official method):")
    print("  from PIL import Image; import numpy as np")
    print("  img = np.array(Image.open('output.png'))")
    print("  print(img.min(), img.max())           # expect 0 ... ~65535")
    print("  depth_cm = img / 65535.0 * 1000.0")
    print("  print(depth_cm.min(), depth_cm.max()) # expect 0 ... ~1000 cm")
