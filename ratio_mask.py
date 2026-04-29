import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .bg_remove import POSITIONS, resolve_position
from .bg_remove_utils import hex_to_rgb

_RATIO_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")


def _parse_ratio(s):
    m = _RATIO_RE.match(s or "")
    if not m:
        return 1.0, 1.0
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return 1.0, 1.0
    return a, b


class RatioMask:
    """
    Generate a white rectangle of a chosen aspect ratio inside a colored
    canvas, fit-resized to the largest size that fits within the canvas
    minus per-side paddings, and placed at one of nine grid positions.

    The MASK output is 1.0 inside the rectangle and 0.0 outside (independent
    of bg_color). The IMAGE output paints the rectangle white over bg_color
    (so default '#000000' gives the classic white-on-black look).

    corner_radius rounds the rectangle's corners (capped at half the shorter
    side). blur applies a Gaussian to the mask edges; the IMAGE is derived
    by blending white over bg_color through that mask, so soft edges show
    consistently in both outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "ratio": ("STRING", {"default": "1:1"}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "bg_color": ("STRING", {"default": "#000000"}),
                "corner_radius": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, width, height, ratio, position, bg_color,
                 corner_radius, blur,
                 padding_top, padding_bottom, padding_left, padding_right):
        rw, rh = _parse_ratio(ratio)
        target_aspect = rw / rh

        eff_w = max(1, width - padding_left - padding_right)
        eff_h = max(1, height - padding_top - padding_bottom)

        if target_aspect >= eff_w / eff_h:
            shape_w = eff_w
            shape_h = max(1, int(round(eff_w / target_aspect)))
        else:
            shape_h = eff_h
            shape_w = max(1, int(round(eff_h * target_aspect)))

        cy, cx = resolve_position(
            position, height, width, shape_h, shape_w,
            padding_top, padding_bottom, padding_left, padding_right,
        )

        pil_mask = Image.new("L", (width, height), 0)
        if shape_w > 0 and shape_h > 0:
            draw = ImageDraw.Draw(pil_mask)
            box = [cx, cy, cx + shape_w - 1, cy + shape_h - 1]
            r = min(int(corner_radius), min(shape_w, shape_h) // 2)
            if r > 0:
                draw.rounded_rectangle(box, radius=r, fill=255)
            else:
                draw.rectangle(box, fill=255)

        if blur > 0:
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=float(blur)))

        mask_np = np.asarray(pil_mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        rgb = hex_to_rgb(bg_color)
        bg_t = torch.tensor(rgb, dtype=torch.float32) / 255.0

        m3 = mask.unsqueeze(-1)
        bg_canvas = torch.zeros((1, height, width, 3), dtype=torch.float32)
        bg_canvas[..., :] = bg_t
        white = torch.ones((1, height, width, 3), dtype=torch.float32)
        image = bg_canvas * (1.0 - m3) + white * m3

        return (image, mask)


NODE_CLASS_MAPPINGS = {"RatioMask": RatioMask}
NODE_DISPLAY_NAME_MAPPINGS = {"RatioMask": "Whisker: Ratio Mask"}
