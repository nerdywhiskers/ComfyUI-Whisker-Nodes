import torch
import torch.nn.functional as F

from .bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox, hex_to_rgb

POSITIONS = [
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


def resolve_position(position, canvas_h, canvas_w, asset_h, asset_w,
                      pad_top, pad_bottom, pad_left, pad_right):
    v, h = position.split("-")
    if v == "top":
        cy = pad_top
    elif v == "bottom":
        cy = canvas_h - asset_h - pad_bottom
    else:
        inner_h = canvas_h - pad_top - pad_bottom
        cy = pad_top + (inner_h - asset_h) // 2
    if h == "left":
        cx = pad_left
    elif h == "right":
        cx = canvas_w - asset_w - pad_right
    else:
        inner_w = canvas_w - pad_left - pad_right
        cx = pad_left + (inner_w - asset_w) // 2
    return cy, cx


class BGRemoveCompose:
    """
    Remove background with BiRefNet/RMBG-2.0 and composite the asset onto a
    canvas of user-specified size.

    - background = 'alpha': transparent canvas, asset alpha preserved.
    - background = 'color': solid bg_color canvas, asset alpha-blended over it.
    - resize_to_fit ON: asset is scaled to fit inside the canvas minus the
      per-side paddings, preserving aspect ratio. The 'scale' slider is
      ignored in this mode.
    - resize_to_fit OFF: asset is sized via the 'scale' multiplier on its
      original (post-RMBG bbox) dimensions.
    - padding_top/right/bottom/left: per-side pixel margins between asset
      and canvas edges. Shift the 9-grid positions inward (e.g., top-left
      places at (padding_top, padding_left)). Center positions center
      within the inner box left after subtracting the paddings.

    Output IMAGE is always 4-channel RGBA. In 'color' mode the alpha channel
    is fully opaque so the result saves identically to a 3-channel PNG.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODEL_REGISTRY.keys()), {"default": "BiRefNet"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "background": (["alpha", "color"], {"default": "alpha"}),
                "bg_color": ("STRING", {"default": "#ffffff"}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "resize_to_fit": ("BOOLEAN", {"default": False}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "compose"
    CATEGORY = "whisker-nodes"

    def compose(self, image, model, width, height, background, bg_color,
                position, resize_to_fit, scale,
                padding_top, padding_bottom, padding_left, padding_right):
        b = image.shape[0]

        out_imgs = torch.zeros((b, height, width, 4), dtype=torch.float32)
        if background == "color":
            rgb = hex_to_rgb(bg_color)
            bg_rgba = torch.tensor([rgb[0], rgb[1], rgb[2], 255], dtype=torch.float32) / 255.0
            out_imgs[..., :] = bg_rgba

        out_masks = torch.zeros((b, height, width), dtype=torch.float32)

        masks = predict_mask(image, model)

        eff_w = max(1, width - padding_left - padding_right)
        eff_h = max(1, height - padding_top - padding_bottom)

        for i in range(b):
            mask_i = masks[i]
            bbox = mask_bbox(mask_i)
            if bbox is None:
                continue
            y0, x0, y1, x1 = bbox
            asset = image[i, y0:y1, x0:x1, :]
            alpha = mask_i[y0:y1, x0:x1]
            ah, aw = asset.shape[:2]

            if resize_to_fit:
                fit_scale = min(eff_w / aw, eff_h / ah)
                new_h = max(1, int(round(ah * fit_scale)))
                new_w = max(1, int(round(aw * fit_scale)))
            else:
                new_h = max(1, int(round(ah * scale)))
                new_w = max(1, int(round(aw * scale)))

            asset_chw = asset.permute(2, 0, 1).unsqueeze(0)
            asset_resized = F.interpolate(asset_chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            asset_resized = asset_resized.squeeze(0).permute(1, 2, 0)
            alpha_resized = F.interpolate(
                alpha.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

            cy, cx = resolve_position(
                position, height, width, new_h, new_w,
                padding_top, padding_bottom, padding_left, padding_right,
            )

            dst_y0 = max(0, cy)
            dst_x0 = max(0, cx)
            dst_y1 = min(height, cy + new_h)
            dst_x1 = min(width, cx + new_w)
            src_y0 = dst_y0 - cy
            src_x0 = dst_x0 - cx
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            src_x1 = src_x0 + (dst_x1 - dst_x0)
            if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                continue

            asset_crop = asset_resized[src_y0:src_y1, src_x0:src_x1, :]
            alpha_crop = alpha_resized[src_y0:src_y1, src_x0:src_x1]
            a3 = alpha_crop.unsqueeze(-1)

            if background == "alpha":
                out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 0:3] = asset_crop
                out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 3] = alpha_crop
            else:
                bg_region = out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 0:3]
                out_imgs[i, dst_y0:dst_y1, dst_x0:dst_x1, 0:3] = asset_crop * a3 + bg_region * (1.0 - a3)

            out_masks[i, dst_y0:dst_y1, dst_x0:dst_x1] = alpha_crop

        return (out_imgs, out_masks)


NODE_CLASS_MAPPINGS = {"BGRemoveCompose": BGRemoveCompose}
NODE_DISPLAY_NAME_MAPPINGS = {"BGRemoveCompose": "Whisker: BG Remove + Compose"}
