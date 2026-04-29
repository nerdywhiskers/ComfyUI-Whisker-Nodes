import torch
import torch.nn.functional as F
from PIL import ImageFilter
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

class OffsetImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "mask_thickness": ("INT", {"default": 10, "min": 1, "max": 512, "step": 1}),
                "mask_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("offset_image", "seam_mask",)
    FUNCTION = "offset_image"
    CATEGORY = "whisker-nodes"

    def offset_image(self, image, offset_x, offset_y, mask_thickness, mask_blur):
        b, h, w, c = image.shape
        image_offset = torch.roll(image, shifts=(offset_y, offset_x), dims=(1, 2))

        mask = torch.zeros((h, w), dtype=torch.float32)
        half_lo = mask_thickness // 2
        half_hi = mask_thickness - half_lo

        if offset_x != 0:
            seam_x = offset_x % w
            x_lo = max(0, seam_x - half_lo)
            x_hi = min(w, seam_x + half_hi)
            mask[:, x_lo:x_hi] = 1.0

        if offset_y != 0:
            seam_y = offset_y % h
            y_lo = max(0, seam_y - half_lo)
            y_hi = min(h, seam_y + half_hi)
            mask[y_lo:y_hi, :] = 1.0

        if mask_blur > 0:
            pil_mask = to_pil_image(mask)
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            mask = to_tensor(pil_mask).squeeze(0)

        mask = mask.unsqueeze(0).repeat(b, 1, 1)
        return (image_offset, mask)
