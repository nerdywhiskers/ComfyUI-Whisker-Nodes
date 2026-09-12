import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .bg_remove import POSITIONS, resolve_canvas_anchor
from ..utils.bg_remove_utils import hex_to_rgb


class ShapeMask:
    """
    Generate a white shape of explicit pixel size (x by y) inside a colored
    canvas, anchored at one of nine positions.

    The shape is clamped to the canvas and edge-anchored (not grid-cell
    centered), so the mask is always fully within the canvas bounds.

    The MASK output is 1.0 inside the shape and 0.0 outside (independent
    of bg_color). The IMAGE output paints the shape white over bg_color
    (so default '#000000' gives the classic white-on-black look).

    corner_radius is a 0-100 slider: percent of the maximum rounding (half
    the shorter side). At 100 the shape is drawn as an inscribed ellipse —
    a circle when x equals y. blur applies a Gaussian to the mask edges;
    the IMAGE is derived by blending white over bg_color through that mask,
    so soft edges show consistently in both outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "canvas height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "x": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Shape width in pixels (clamped to the canvas)."}),
                "y": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Shape height in pixels (clamped to the canvas)."}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "bg_color": ("STRING", {"default": "#000000"}),
                "corner_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Corner rounding as a percent of the maximum (half the shorter side). 100 draws an inscribed ellipse — a circle when x equals y."}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, **kwargs):
        # Input names contain spaces, so they arrive via **kwargs.
        canvas_width = int(kwargs["canvas width"])
        canvas_height = int(kwargs["canvas height"])
        x = int(kwargs["x"])
        y = int(kwargs["y"])
        position = kwargs.get("position", "middle-center")
        bg_color = kwargs.get("bg_color", "#000000")
        corner_radius = float(kwargs.get("corner_radius", 0.0))
        blur = float(kwargs.get("blur", 0.0))

        # Clamp the shape to the canvas so it always fits, then edge-anchor
        # it: the mask can never leave the canvas bounds.
        shape_w = max(1, min(x, canvas_width))
        shape_h = max(1, min(y, canvas_height))

        cy, cx = resolve_canvas_anchor(
            position, canvas_height, canvas_width, shape_h, shape_w,
            0, 0, 0, 0,
        )

        pil_mask = Image.new("L", (canvas_width, canvas_height), 0)
        draw = ImageDraw.Draw(pil_mask)
        box = [cx, cy, cx + shape_w - 1, cy + shape_h - 1]
        if corner_radius >= 100:
            draw.ellipse(box, fill=255)
        else:
            r = int(round(max(0.0, corner_radius) / 100.0 * min(shape_w, shape_h) / 2))
            r = min(r, min(shape_w, shape_h) // 2)
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
        bg_canvas = torch.zeros((1, canvas_height, canvas_width, 3), dtype=torch.float32)
        bg_canvas[..., :] = bg_t
        white = torch.ones((1, canvas_height, canvas_width, 3), dtype=torch.float32)
        image = bg_canvas * (1.0 - m3) + white * m3

        return (image, mask)


NODE_CLASS_MAPPINGS = {
    "shape_mask": ShapeMask,
    # Legacy id so workflows saved with the old Ratio Mask node still load.
    "ratio_mask": ShapeMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {"shape_mask": "Whisker: Shape Mask"}
