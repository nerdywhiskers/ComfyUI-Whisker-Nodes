import numpy as np
import torch
from PIL import Image, ImageFilter


def _smooth_noise_1d(length, scale, seed):
    """
    Smoothstep-interpolated 1D noise of the given length. Output range ~[-1, 1].
    """
    scale = max(1.0, float(scale))
    rng = np.random.default_rng(seed)
    n_keys = max(2, int(np.ceil(length / scale)) + 2)
    keys = rng.uniform(-1.0, 1.0, n_keys)

    t = np.arange(length) / scale
    idx = np.floor(t).astype(np.int64)
    frac = t - idx
    idx = np.clip(idx, 0, n_keys - 2)
    smooth = frac * frac * (3.0 - 2.0 * frac)
    a = keys[idx]
    b = keys[idx + 1]
    return (a + (b - a) * smooth).astype(np.float32)


def _split_sizes(total, num, primary_idx, primary_size):
    """
    Return a list of `num` sizes summing exactly to `total`. The strip at
    primary_idx (0-indexed) takes primary_size pixels (clamped to [0, total]);
    the remainder splits evenly across the others, with the rounding remainder
    distributed 1px at a time across non-primary strips.
    """
    primary_idx = max(0, min(num - 1, primary_idx))
    primary_size = max(0, min(total, primary_size))

    if num == 1:
        return [total]

    remaining = total - primary_size
    other_count = num - 1
    base = max(0, remaining // other_count)
    rem = remaining - base * other_count

    sizes = [base] * num
    sizes[primary_idx] = primary_size

    j = 0
    for i in range(num):
        if i == primary_idx:
            continue
        if j < rem:
            sizes[i] += 1
        j += 1
    return sizes


class StripMaskGenerator:
    """
    Generate 2-8 strip masks that tile a container without gaps or overlap.

    One strip is sized explicitly (primary_mask_size at primary_mask_index);
    the others split the remaining space evenly. Strips run either as
    horizontal bands (full width, varying height) or vertical bands (full
    height, varying width).

    Interior boundaries are displaced by a single shared smooth-1D-noise
    vector along the perpendicular axis, so all edges warp coherently as
    one bent stack (strip thicknesses preserved locally).

    8 MASK outputs are always returned. Slots beyond num_masks are zeros.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "num_masks": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "primary_mask_index": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "primary_mask_size": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "noise_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 1.0}),
                "noise_scale": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1024.0, "step": 1.0}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",) * 8
    RETURN_NAMES = tuple(f"mask_{i + 1}" for i in range(8))
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, width, height, num_masks, orientation,
                 primary_mask_index, primary_mask_size,
                 noise_amount, noise_scale, blur, seed):
        if orientation == "horizontal":
            total = height
            edge_length = width
        else:
            total = width
            edge_length = height

        sizes = _split_sizes(total, num_masks, primary_mask_index - 1, primary_mask_size)
        boundaries = np.cumsum([0] + sizes)

        if noise_amount > 0:
            shared_noise = _smooth_noise_1d(edge_length, noise_scale, seed) * float(noise_amount)
        else:
            shared_noise = np.zeros(edge_length, dtype=np.float32)

        y_grid = np.arange(height, dtype=np.float32).reshape(-1, 1)
        x_grid = np.arange(width, dtype=np.float32).reshape(1, -1)

        masks = []
        for i in range(num_masks):
            if orientation == "horizontal":
                top = np.zeros(edge_length, dtype=np.float32) if i == 0 else (boundaries[i] + shared_noise)
                bottom = np.full(edge_length, total, dtype=np.float32) if i == num_masks - 1 else (boundaries[i + 1] + shared_noise)
                mask = ((y_grid >= top[np.newaxis, :]) & (y_grid < bottom[np.newaxis, :])).astype(np.float32)
            else:
                left = np.zeros(edge_length, dtype=np.float32) if i == 0 else (boundaries[i] + shared_noise)
                right = np.full(edge_length, total, dtype=np.float32) if i == num_masks - 1 else (boundaries[i + 1] + shared_noise)
                mask = ((x_grid >= left[:, np.newaxis]) & (x_grid < right[:, np.newaxis])).astype(np.float32)
            if blur > 0:
                pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur)))
                mask = np.asarray(pil, dtype=np.float32) / 255.0
            masks.append(mask)

        outputs = []
        for i in range(8):
            if i < num_masks:
                t = torch.from_numpy(masks[i]).unsqueeze(0)
            else:
                t = torch.zeros((1, height, width), dtype=torch.float32)
            outputs.append(t)
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {"StripMaskGenerator": StripMaskGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"StripMaskGenerator": "Whisker: Strip Mask Generator"}
