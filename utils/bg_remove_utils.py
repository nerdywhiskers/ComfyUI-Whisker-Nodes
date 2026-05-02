import torch
import torch.nn.functional as F
from torchvision import transforms

# SECURITY WARNING: trust_remote_code=True allows executing arbitrary code from HuggingFace.
# Only use models from trusted sources. See: https://huggingface.co/docs/hub/security
MODEL_REGISTRY = {
    "BiRefNet": "ZhengPeng7/BiRefNet",
    "RMBG-2.0": "briaai/RMBG-2.0",
}

MODEL_CACHE = {}


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
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_REGISTRY[name], trust_remote_code=True
        )
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
    model.to(offload).eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    MODEL_CACHE[name] = model
    return model


@torch.inference_mode()
def predict_mask(image_bhwc, model_name):
    """
    image_bhwc: torch.Tensor in ComfyUI IMAGE format (B, H, W, 3), float32 in [0, 1].
    Returns: mask (B, H, W) float32 in [0, 1], at the original H×W.

    The model lives on the offload device between calls and is moved to the
    compute device only for inference, then moved back. This lets BiRefNet /
    RMBG-2.0 share VRAM with diffusion models on smaller GPUs.
    """
    model = load_model(model_name)
    device, offload = _devices()

    _, h, w, _ = image_bhwc.shape
    img_bchw = image_bhwc.permute(0, 3, 1, 2).contiguous()

    pre = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resized = F.interpolate(img_bchw, size=(1024, 1024), mode="bilinear", align_corners=False)

    model.to(device)
    try:
        model_dtype = next(model.parameters()).dtype
        inp = pre(resized).to(device=device, dtype=model_dtype)
        preds = model(inp)[-1].float().sigmoid()
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        mask = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        result = mask.squeeze(1).clamp(0.0, 1.0).cpu()
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
