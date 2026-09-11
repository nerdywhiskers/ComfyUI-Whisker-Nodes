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


def resolve_canvas_anchor(
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
    """Place an asset against an edge or center of the padded canvas area."""
    inner_y0 = min(pad_top, canvas_h - 1)
    inner_x0 = min(pad_left, canvas_w - 1)
    inner_y1 = max(inner_y0 + 1, canvas_h - pad_bottom)
    inner_x1 = max(inner_x0 + 1, canvas_w - pad_right)
    vertical, horizontal = position.split("-")
    y = (
        inner_y0
        if vertical == "top"
        else inner_y1 - asset_h
        if vertical == "bottom"
        else inner_y0 + (inner_y1 - inner_y0 - asset_h) // 2
    )
    x = (
        inner_x0
        if horizontal == "left"
        else inner_x1 - asset_w
        if horizontal == "right"
        else inner_x0 + (inner_x1 - inner_x0 - asset_w) // 2
    )
    return y, x


class BGRemoveCompose:
    """
    Remove background with BiRefNet/RMBG-2.0 and composite the asset onto a
    canvas of user-specified size.

    Pipeline:
      1. Predict foreground mask.
      2. Compute tight bounding box of the mask.
      3. Expand the bbox by crop_padding on all sides.
      4. Crop the image + alpha to that expanded region.
      5. Resize the cropped asset into the padded destination area.
      6. Anchor it to an edge or the center of that area.

    - resize_to_fit: fit the crop into the destination area proportionally.
      When disabled, apply scale and proportionally reduce only if needed.
    - crop_padding: extra pixels added to all sides of the mask bbox before
      cropping.  Useful to give the subject breathing room.
    - position: simple anchor — places the resized asset flush against the
      named edge/corner of the padded area (e.g. 'bottom-center' is centered
      horizontally and sits against the bottom margin).
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
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "padding_top": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "padding_bottom": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "padding_left": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "padding_right": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "crop_padding": (
                    "INT",
                    {"default": 20, "min": 0, "max": 4096, "step": 1},
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
        position,
        resize_to_fit,
        scale,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
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
        inner_h = max(1, height - padding_top - padding_bottom)
        inner_w = max(1, width - padding_left - padding_right)

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

            fit_scale = min(inner_w / aw, inner_h / ah)
            resize_scale = fit_scale if resize_to_fit else min(scale, fit_scale)
            new_w = min(inner_w, max(1, int(round(aw * resize_scale))))
            new_h = min(inner_h, max(1, int(round(ah * resize_scale))))

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

            dst_y, dst_x = resolve_canvas_anchor(
                position,
                height,
                width,
                new_h,
                new_w,
                padding_top,
                padding_bottom,
                padding_left,
                padding_right,
            )

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
