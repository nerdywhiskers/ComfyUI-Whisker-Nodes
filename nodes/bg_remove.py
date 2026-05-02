import torch
import torch.nn.functional as F

from ..utils.bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox, hex_to_rgb

POSITIONS = [
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


def resolve_position(position, canvas_h, canvas_w, asset_h, asset_w,
                      pad_top, pad_bottom, pad_left, pad_right):
    """
    Divide the canvas (minus per-side padding) into a 3x3 grid and center the
    asset within the named cell. e.g. 'top-left' centers the asset in the
    top-left third of the screen, not flush against the top-left corner.
    Padding shrinks the active region; cells split that inner region evenly.
    """
    v, h = position.split("-")
    v_idx = {"top": 0, "middle": 1, "bottom": 2}[v]
    h_idx = {"left": 0, "center": 1, "right": 2}[h]

    inner_h = max(1, canvas_h - pad_top - pad_bottom)
    inner_w = max(1, canvas_w - pad_left - pad_right)
    cell_h = inner_h / 3
    cell_w = inner_w / 3

    cell_y0 = pad_top + v_idx * cell_h
    cell_x0 = pad_left + h_idx * cell_w
    cy = int(round(cell_y0 + (cell_h - asset_h) / 2))
    cx = int(round(cell_x0 + (cell_w - asset_w) / 2))
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
    - position: 9-cell grid semantic. The canvas (minus paddings) is split
      into a 3x3 grid and the asset is centered within the named cell. So
      'top-left' centers the asset within the top-left third of the canvas,
      not flush against the corner. Use small scales / resize_to_fit=False
      to actually see the regional placement.
    - padding_top/right/bottom/left: per-side pixel margins that shrink the
      active region before it is split into the 3x3 grid.

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
