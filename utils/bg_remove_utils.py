import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

# SECURITY WARNING: trust_remote_code=True allows executing arbitrary code from HuggingFace.
# Only use models from trusted sources. See: https://huggingface.co/docs/hub/security
MODEL_REGISTRY = {
    "BiRefNet": "ZhengPeng7/BiRefNet",
    "RMBG-2.0": "briaai/RMBG-2.0",
}

# Local subdir names to probe under each ComfyUI `models/rmbg` root.
LOCAL_SUBDIRS = {
    "BiRefNet": ("BiRefNet", "birefnet", "ZhengPeng7--BiRefNet"),
    "RMBG-2.0": ("RMBG-2.0", "rmbg-2.0", "RMBG-2_0", "briaai--RMBG-2.0"),
}

MODEL_CACHE = {}

# Frames per bg-removal forward pass when the caller doesn't specify one.
# The model upsamples every frame to 1024x1024, so VRAM scales with batch
# size — chunking keeps high frame counts from OOMing.
DEFAULT_MASK_BATCH_SIZE = 4


def _rmbg_roots():
    """Candidate `models/rmbg` directories (ComfyUI portable + extras)."""
    roots = []
    try:
        import folder_paths  # type: ignore

        models_dir = Path(getattr(folder_paths, "models_dir", ""))
        if str(models_dir) not in ("", "."):
            roots.append(models_dir / "rmbg")
    except Exception:
        pass
    try:
        # <ComfyUI>/custom_nodes/ComfyUI-Whisker-Nodes/utils -> <ComfyUI>
        comfy_dir = Path(__file__).resolve().parents[3]
        roots.append(comfy_dir / "models" / "rmbg")
    except Exception:
        pass
    for env_key in ("COMFYUI_MODELS_DIR", "COMFY_MODELS_DIR"):
        val = os.environ.get(env_key)
        if val:
            roots.append(Path(val) / "rmbg")
    # Extra path observed on this machine (extra_model_paths.yaml).
    for p in (Path(r"F:\comfymodels\models\rmbg"), Path(r"F:\ComfyUI_windows_portable\ComfyUI\models\rmbg")):
        roots.append(p)
    # De-dupe while preserving order.
    seen, unique = set(), []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _local_model_path(name):
    """Return a local model dir with config + weights, or None."""
    for root in _rmbg_roots():
        for sub in LOCAL_SUBDIRS.get(name, (name,)):
            d = root / sub
            try:
                if not d.is_dir():
                    continue
                has_config = (d / "config.json").is_file() and (d / "config.json").stat().st_size > 0
                has_weights = any(
                    (d / f).is_file() and (d / f).stat().st_size > 0
                    for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
                )
                if has_config and has_weights:
                    return str(d)
            except Exception:
                continue
    return None


def _hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _devices():
    """Return (compute_device, offload_device), preferring ComfyUI's manager."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device(), mm.unet_offload_device()
    except Exception:
        compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return compute, torch.device("cpu")


def _soft_empty_cache():
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError as e:
        raise ImportError(
            "The 'transformers' package is required for the BG removal nodes. "
            "Install it in your ComfyUI environment:\n"
            "  Portable: python_embeded\\python.exe -m pip install transformers\n"
            "  System:   pip install transformers"
        ) from e

    _, offload = _devices()
    repo_id = MODEL_REGISTRY[name]
    # Prefer local ComfyUI weights (avoids gated HF downloads, e.g. RMBG-2.0
    # already shipped by ComfyUI-RMBG under models/rmbg/RMBG-2.0).
    source = _local_model_path(name) or repo_id
    kwargs = {"trust_remote_code": True}
    token = _hf_token()
    if token and source == repo_id:
        kwargs["token"] = token
    try:
        # BiRefNet/RMBG-2.0's custom modelling does `self.config = Config()`
        # (plain training hparams, birefnet.py:1984), discarding the HF
        # PretrainedConfig. transformers>=4.55 calls
        # `config.get_text_config()` inside `tie_weights()` during
        # `from_pretrained`, which crashes with
        # `AttributeError: 'Config' object has no attribute 'get_text_config'`.
        # The model has no tied embeddings, so no-op the hook for this load.
        try:
            from transformers.modeling_utils import PreTrainedModel as _PTM

            _orig_tie = _PTM.tie_weights
            _PTM.tie_weights = lambda self: None  # type: ignore[method-assign]
        except Exception:
            _PTM = None  # type: ignore[assignment]
            _orig_tie = None
        try:
            model = AutoModelForImageSegmentation.from_pretrained(source, **kwargs)
        finally:
            try:
                if _PTM is not None and _orig_tie is not None:
                    _PTM.tie_weights = _orig_tie  # type: ignore[method-assign]
            except Exception:
                pass
    except ImportError as e:
        if "timm" in str(e) or "timm.layers" in str(e):
            raise ImportError(
                "The BiRefNet/RMBG model requires 'timm>=1.0' (older versions "
                "lack the 'timm.layers' module). Upgrade in the SAME Python that "
                "runs ComfyUI:\n"
                "  Portable: python_embeded\\python.exe -m pip install --upgrade \"timm>=1.0\"\n"
                "  System:   pip install --upgrade \"timm>=1.0\"\n"
                "If you already installed timm and still see this, your install "
                "likely went to a different Python (e.g. user site-packages) than "
                "the one ComfyUI launches. Run the upgrade with the exact "
                "interpreter shown in the ComfyUI startup log."
            ) from e
        raise
    except OSError as e:
        msg = str(e)
        if "gated repo" in msg or "restricted" in msg or "403" in msg:
            raise OSError(
                f"Model '{name}' ({repo_id}) is gated on HuggingFace and this "
                "machine is not authenticated.\n"
                f"  1. Visit https://huggingface.co/{repo_id} and accept the terms.\n"
                "  2. Create a token at https://huggingface.co/settings/tokens, then:\n"
                "       python_embeded\\python.exe -m huggingface_hub.commands.huggingface_cli login\n"
                "     or set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) before launching ComfyUI.\n"
                "  3. Retry the node.\n"
                f"Local fallback also checked under models/rmbg/{name} "
                f"(last tried: {source}). Place config.json + model.safetensors there "
                "to run fully offline."
            ) from e
        if "not a valid JSON file" in msg:
            raise OSError(
                f"Corrupted HuggingFace cache for '{name}' ({repo_id}): {msg}\n"
                "A 0-byte config.json / model file was left by an interrupted download.\n"
                "  1. Close ComfyUI.\n"
                "  2. Delete the cache folder, e.g.:\n"
                f"       Remove-Item -Recurse -Force \"$env:HF_HUB_CACHE\\models--{repo_id.replace('/', '--')}\"\n"
                "     (your HF_HUB_CACHE is F:\\hf-cache\\hub)\n"
                "  3. Relaunch and retry — it will re-download cleanly."
            ) from e
        raise
    model.to(offload).eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    MODEL_CACHE[name] = model
    return model


@torch.inference_mode()
def predict_mask(image_bhwc, model_name, batch_size=None):
    """
    image_bhwc: torch.Tensor in ComfyUI IMAGE format (B, H, W, 3 or 4), float32 in [0, 1].
    Returns: mask (B, H, W) float32 in [0, 1], at the original H×W.

    RGBA input (e.g. from Sprite Sheet Generator, which always emits 4
    channels) is accepted: only the first 3 channels are fed to the model.
    Callers that need to preserve incoming transparency should combine the
    returned mask with image_bhwc[..., 3] themselves.

    batch_size caps how many frames go through the model per forward pass
    (default DEFAULT_MASK_BATCH_SIZE). Only one chunk is on the compute
    device at a time, so large batches cost time, not VRAM. On a CUDA out
    of memory, the failing chunk is automatically retried in halves down
    to single frames before giving up with guidance.

    The model lives on the offload device between calls and is moved to the
    compute device only for inference, then moved back. This lets BiRefNet /
    RMBG-2.0 share VRAM with diffusion models on smaller GPUs.
    """
    model = load_model(model_name)
    device, offload = _devices()

    per = max(1, int(batch_size or DEFAULT_MASK_BATCH_SIZE))

    _, h, w, c = image_bhwc.shape
    if c == 4:
        image_bhwc = image_bhwc[..., :3]
    elif c != 3:
        raise ValueError(f"Expected IMAGE with 3 or 4 channels, got shape {tuple(image_bhwc.shape)}")
    img_bchw = image_bhwc.permute(0, 3, 1, 2).contiguous()

    pre = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resized = F.interpolate(img_bchw, size=(1024, 1024), mode="bilinear", align_corners=False)

    def _oom_guidance():
        return (
            "BG-removal model ran out of memory even one frame at a time. "
            "Lower 'target_frame_count', set 'target_resolution' (e.g. 2048), "
            "close other GPU apps, or after the diffusion model has offloaded, then retry."
        )

    model.to(device)
    try:
        model_dtype = next(model.parameters()).dtype
        out = []

        def run(lo, hi):
            n = hi - lo
            try:
                inp = pre(resized[lo:hi]).to(device=device, dtype=model_dtype)
                preds = model(inp)[-1].float().sigmoid()
                if preds.dim() == 3:
                    preds = preds.unsqueeze(1)
                m = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
                out.append(m.squeeze(1).clamp(0.0, 1.0).cpu())
                del inp, preds, m
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                _soft_empty_cache()
                if n <= 1:
                    raise RuntimeError(_oom_guidance()) from e
                mid = lo + n // 2
                run(lo, mid)
                run(mid, hi)
            else:
                _soft_empty_cache()

        lo, total = 0, resized.shape[0]
        while lo < total:
            run(lo, min(lo + per, total))
            lo += per
        result = torch.cat(out, dim=0)
    finally:
        model.to(offload)
        _soft_empty_cache()

    return result


def mask_bbox(mask_hw, threshold=0.05):
    """
    Compute tight bounding box of non-zero region in a single (H, W) mask.
    Returns (y0, x0, y1, x1) inclusive-exclusive, or None if mask is empty.
    """
    binary = mask_hw > threshold
    if not binary.any():
        return None
    rows = binary.any(dim=1)
    cols = binary.any(dim=0)
    y0 = int(rows.float().argmax().item())
    y1 = int(len(rows) - rows.flip(0).float().argmax().item())
    x0 = int(cols.float().argmax().item())
    x1 = int(len(cols) - cols.flip(0).float().argmax().item())
    return y0, x0, y1, x1


def hex_to_rgb(hex_str):
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
