"""
=============================================================
  setup_dav2.py — Run this ONCE at the top of your Kaggle
  notebook before running train.py or run.py
=============================================================
"""
import os
import subprocess
import sys


def run(cmd, check=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


# ── 1. Install timm (required by DAV2 ViT backbone) ───────────────────────────
run("pip install timm --quiet")

# ── 2. Clone the official Depth Anything V2 repository ────────────────────────
DAV2_DIR = "/kaggle/working/Depth-Anything-V2"
if not os.path.exists(DAV2_DIR):
    run(f"git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 {DAV2_DIR}")
else:
    print(f"[Setup] DAV2 repo already exists at {DAV2_DIR}, skipping clone.")

# Install it as an editable package so 'import depth_anything_v2' works
run(f"pip install -e {DAV2_DIR} --quiet")

# ── 3. Download the metric weights (ViT-L, Hypersim) ──────────────────────────
WEIGHTS_DIR  = "/kaggle/working/NTIRE-HR_Depth-DVision/checkpoints_new/pretrained"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "depth_anything_v2_metric_hypersim_vitl.pth")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.exists(WEIGHTS_FILE):
    print("\n[Setup] Downloading DAV2 metric weights (~1.4 GB)...")
    run(
        f"huggingface-cli download "
        f"depth-anything/Depth-Anything-V2-Metric-Hypersim-Large "
        f"depth_anything_v2_metric_hypersim_vitl.pth "
        f"--local-dir {WEIGHTS_DIR}"
    )
else:
    print(f"[Setup] Weights already present at {WEIGHTS_FILE}, skipping download.")

# ── 4. Quick sanity check ──────────────────────────────────────────────────────
print("\n[Setup] Verifying import…")
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
    print("[Setup] ✓  'depth_anything_v2' imported successfully!")
    print("[Setup] ✓  Native DAV2 is now active — train.py will use metric depth output.")
except ImportError as e:
    print(f"[Setup] ✗  Import FAILED: {e}")
    print("         Please restart the kernel and re-run this cell.")

print(f"\n[Setup] Done. Weights path to pass to train.py:")
print(f"  --pretrained_weights {WEIGHTS_FILE}")
