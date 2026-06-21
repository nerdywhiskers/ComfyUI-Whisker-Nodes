import torch
import torch.nn.functional as F

from ..utils.bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox, hex_to_rgb

POSITIONS = [
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "middle-center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]


def resolve_position(
    position,
    canvas_h,
    canvas_w,
    asset_h,
    asset_w,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
):
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


def resolve_anchor(position, canvas_w, canvas_h, asset_w, asset_h):
    anchors = {
        "top-left": (0, 0),
        "top-center": ((canvas_w - asset_w) // 2, 0),
        "top-right": (canvas_w - asset_w, 0),
        "middle-left": (0, (canvas_h - asset_h) // 2),
        "middle-center": ((canvas_w - asset_w) // 2, (canvas_h - asset_h) // 2),
        "middle-right": (canvas_w - asset_w, (canvas_h - asset_h) // 2),
        "bottom-left": (0, canvas_h - asset_h),
        "bottom-center": ((canvas_w - asset_w) // 2, canvas_h - asset_h),
        "bottom-right": (canvas_w - asset_w, canvas_h - asset_h),
    }
    return anchors.get(position, anchors["middle-center"])


class BGRemoveCompose:
    """
    Remove background with BiRefNet/RMBG-2.0 and composite the asset onto a
    canvas of user-specified size.

    Pipeline:
      1. Predict foreground mask.
      2. Compute tight bounding box of the mask.
      3. Expand the bbox by crop_padding on all sides.
      4. Crop the image + alpha to that expanded region.
      5. Resize the cropped asset into the canvas (fit-to-canvas or scale).
      6. Anchor at the chosen position (never out-of-bounds because the
         resized asset is clamped to canvas dimensions).

    - resize_mode = 'fit': scale the cropped asset to fit within the canvas
      while preserving aspect ratio (letterbox effect). The padding area
      is filled by the background setting.
    - resize_mode = 'scale': multiply the cropped asset dimensions by the
      scale factor. The result is clamped to the canvas size.
    - crop_padding: extra pixels added to all sides of the mask bbox before
      cropping.  Useful to give the subject breathing room.
    - position: simple anchor — places the resized asset flush against the
      named edge/corner of the canvas (e.g. 'bottom-center' = centered
      horizontally, sitting on the bottom edge).
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
                "resize_mode": (["fit", "scale"], {"default": "fit"}),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "position": (POSITIONS, {"default": "middle-center"}),
                "crop_padding": (
                    "INT",
                    {"default": 0, "min": 0, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "compose"
    CATEGORY = "whisker-nodes"

    def compose(
        self,
        image,
        model,
        width,
        height,
        background,
        bg_color,
        resize_mode,
        scale,
        position,
        crop_padding,
    ):
        b = image.shape[0]

        out_imgs = torch.zeros((b, height, width, 4), dtype=torch.float32)
        if background == "color":
            rgb = hex_to_rgb(bg_color)
            bg_rgba = (
                torch.tensor([rgb[0], rgb[1], rgb[2], 255], dtype=torch.float32) / 255.0
            )
            out_imgs[..., :] = bg_rgba

        out_masks = torch.zeros((b, height, width), dtype=torch.float32)
        masks = predict_mask(image, model)

        orig_h, orig_w = image.shape[1], image.shape[2]

        for i in range(b):
            mask_i = masks[i]
            bbox = mask_bbox(mask_i)
            if bbox is None:
                continue

            y0, x0, y1, x1 = bbox

            y0 = max(0, y0 - crop_padding)
            x0 = max(0, x0 - crop_padding)
            y1 = min(orig_h, y1 + crop_padding)
            x1 = min(orig_w, x1 + crop_padding)

            if y1 <= y0 or x1 <= x0:
                continue

            asset = image[i, y0:y1, x0:x1, :]
            alpha = mask_i[y0:y1, x0:x1]
            ah, aw = asset.shape[:2]

            if resize_mode == "fit":
                fit = min(width / aw, height / ah)
                new_w = max(1, int(round(aw * fit)))
                new_h = max(1, int(round(ah * fit)))
            else:
                new_w = max(1, int(round(aw * scale)))
                new_h = max(1, int(round(ah * scale)))

            new_w = min(new_w, width)
            new_h = min(new_h, height)

            asset_chw = asset.permute(2, 0, 1).unsqueeze(0)
            asset_resized = F.interpolate(
                asset_chw, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            asset_resized = asset_resized.squeeze(0).permute(1, 2, 0)
            alpha_resized = (
                F.interpolate(
                    alpha.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .clamp(0.0, 1.0)
            )

            dst_x, dst_y = resolve_anchor(position, width, height, new_w, new_h)

            a3 = alpha_resized.unsqueeze(-1)
            if background == "alpha":
                out_imgs[i, dst_y : dst_y + new_h, dst_x : dst_x + new_w, 0:3] = (
                    asset_resized
                )
                out_imgs[i, dst_y : dst_y + new_h, dst_x : dst_x + new_w, 3] = (
                    alpha_resized
                )
            else:
                bg_region = out_imgs[
                    i, dst_y : dst_y + new_h, dst_x : dst_x + new_w, 0:3
                ]
                out_imgs[i, dst_y : dst_y + new_h, dst_x : dst_x + new_w, 0:3] = (
                    asset_resized * a3 + bg_region * (1.0 - a3)
                )

            out_masks[i, dst_y : dst_y + new_h, dst_x : dst_x + new_w] = alpha_resized

        return (out_imgs, out_masks)


NODE_CLASS_MAPPINGS = {"BGRemoveCompose": BGRemoveCompose}
NODE_DISPLAY_NAME_MAPPINGS = {"BGRemoveCompose": "Whisker: BG Remove + Compose"}
