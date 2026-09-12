import torch
import torch.nn.functional as F

from ..utils.bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox
from .bg_remove import POSITIONS, resolve_canvas_anchor


def _pad_frame(frame, alpha, pad_top, pad_bottom, pad_left, pad_right,
               position="middle-center", crop_padding=0,
               fit_to_canvas=False, original_image_scale=1.0):
    """
    Crop the asset using the mask bbox expanded by crop_padding, scale it
    into (cell - paddings) preserving aspect ratio, and anchor it inside
    the padded area per position. Same crop -> scale -> anchor pipeline as
    BG Remove + Compose, applied per sprite cell. Returns a frame and alpha
    of the same shape as the input frame.

    Defaults (crop 0, no fit-to-canvas, scale 1.0) reduce to the legacy
    behavior: tight bbox, never upscale, centered.
    """
    H, W = int(frame.shape[0]), int(frame.shape[1])
    bbox = mask_bbox(alpha)
    if bbox is None:
        return torch.zeros_like(frame), torch.zeros_like(alpha)

    y0, x0, y1, x1 = bbox

    y0 = max(0, y0 - crop_padding)
    x0 = max(0, x0 - crop_padding)
    y1 = min(H, y1 + crop_padding)
    x1 = min(W, x1 + crop_padding)

    if y1 <= y0 or x1 <= x0:
        return torch.zeros_like(frame), torch.zeros_like(alpha)

    asset = frame[y0:y1, x0:x1, :]
    asset_a = alpha[y0:y1, x0:x1]
    ah, aw = int(asset.shape[0]), int(asset.shape[1])

    avail_h = max(1, H - pad_top - pad_bottom)
    avail_w = max(1, W - pad_left - pad_right)

    fit_scale = min(avail_w / aw, avail_h / ah)
    resize_scale = fit_scale if fit_to_canvas else min(original_image_scale, fit_scale)
    new_w = min(avail_w, max(1, int(round(aw * resize_scale))))
    new_h = min(avail_h, max(1, int(round(ah * resize_scale))))

    if new_h != ah or new_w != aw:
        asset = F.interpolate(
            asset.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).contiguous()
        asset_a = F.interpolate(
            asset_a.unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    cy, cx = resolve_canvas_anchor(
        position, H, W, new_h, new_w,
        pad_top, pad_bottom, pad_left, pad_right,
    )

    new_frame = torch.zeros_like(frame)
    new_alpha = torch.zeros_like(alpha)
    new_frame[cy:cy + new_h, cx:cx + new_w, :] = asset
    new_alpha[cy:cy + new_h, cx:cx + new_w] = asset_a
    return new_frame, new_alpha


def _prune_frames(frames, target_count, start_index, end_index):
    """
    Prune a batch of frames by skipping every N to approach target_count, then
    apply start_index/end_index to the pruned set. end_index = -1 means last.
    Returns the final batch tensor (may be empty).
    """
    total = int(frames.shape[0])
    if total == 0 or target_count < 1:
        return frames[:0]

    step = max(1, total // target_count)
    kept = list(range(0, total, step))[:target_count]
    pruned = frames[kept]

    n = pruned.shape[0]
    if n == 0:
        return pruned

    end = (n - 1) if end_index == -1 else min(end_index, n - 1)
    start = max(0, min(start_index, n - 1))
    if end < start:
        end = start
    return pruned[start:end + 1]


class SpriteSheetGenerator:
    """
    Concatenate frames from a video-like IMAGE batch into a single sprite sheet.

    Pruning order: first reduce to target_frame_count by step-skipping
    (step = total // target_frame_count), then keep [start_index..end_index]
    of the pruned list (end_index = -1 means last).

    target_resolution constrains the final sheet's longest side. When > 0,
    each frame is resized first so the assembled sheet's longest side equals
    this value (bg removal then runs at the resized resolution, which can
    avoid OOM on very large inputs). When 0, frames keep their native size
    and the sheet is just frame_size * grid.

    bg_removal:
      - 'none': sprite sheet alpha is 1.0 everywhere.
      - 'per-frame': run BiRefNet/RMBG-2.0 on each kept frame at the
        (post-resize) frame resolution, then tile the bg-removed frames.
      - 'whole-sheet': assemble the sheet first, then run a single bg
        removal pass on the entire sheet (model downsamples to 1024
        internally, so per-frame mask quality is reduced for large sheets).

    padding_top/bottom/left/right + position anchor each frame's asset:
    bbox-crop expanded by crop_padding via the mask (predicted, or incoming
    alpha when bg_removal is 'none'), scale into (cell - paddings) per
    fit_to_canvas / original_image_scale preserving aspect ratio, and anchor
    per position within the padded area. Same crop -> scale -> anchor
    pipeline as BG Remove + Compose, applied per sprite cell.
    Effective when bg_removal is 'per-frame', and when bg_removal is 'none'
    with RGBA input frames. Ignored for 'whole-sheet'.

    batch_size caps how many frames go through bg removal per forward pass.
    Only one chunk sits on GPU at a time, so high frame counts cost time,
    not VRAM; OOMs halve the chunk automatically down to single frames.

    The output IMAGE is always 4-channel RGBA. The MASK output mirrors the
    sheet's alpha channel (all 1s when bg_removal is 'none' and the input
    has no alpha).

    RGBA input frames are accepted: only RGB is fed to the bg-removal
    model, while any incoming alpha is preserved (multiplied with the
    predicted mask, or tiled directly when bg_removal is 'none') so that
    chaining stays transparent.

    If grid_cols * grid_rows exceeds the final frame count, trailing cells
    are blank. If it is smaller, extra frames are dropped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "target_frame_count": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "end_index": ("INT", {"default": -1, "min": -1, "max": 1024, "step": 1}),
                "grid_cols": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "grid_rows": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "target_resolution": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "bg_removal": (["none", "per-frame", "whole-sheet"], {"default": "none"}),
                "model": (list(MODEL_REGISTRY.keys()), {"default": "BiRefNet"}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "position": (POSITIONS, {"default": "middle-center"}),
                "crop_padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Extra pixels kept around each frame's mask bbox before cropping, for breathing room. Same as BG Remove + Compose.",
                    },
                ),
                "fit_to_canvas": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Fit each cropped asset into the cell proportionally (allows upscaling). When off, the asset keeps its scale and is only reduced to fit. Same as BG Remove + Compose.",
                    },
                ),
                "original_image_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scale factor applied to each cropped asset. Ignored while fit to canvas is enabled. Same as BG Remove + Compose.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Frames per bg-removal forward pass. Only one chunk is on GPU at a time, so lower to 1-2 on small GPUs if per-frame bg removal runs out of memory.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, frames, target_frame_count, start_index, end_index,
                 grid_cols, grid_rows, target_resolution, bg_removal, model,
                 padding_top, padding_bottom, padding_left, padding_right,
                 position="middle-center", batch_size=4,
                 crop_padding=0, fit_to_canvas=False, original_image_scale=1.0):
        final = _prune_frames(frames, target_frame_count, start_index, end_index)
        n_final = int(final.shape[0])

        if n_final == 0:
            empty_img = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return (empty_img, empty_mask)

        if final.shape[3] == 4:
            input_alpha = final[..., 3].contiguous()
            final = final[..., :3].contiguous()
        elif final.shape[3] == 3:
            input_alpha = None
        else:
            raise ValueError(
                f"Expected frames IMAGE with 3 or 4 channels, got shape {tuple(final.shape)}"
            )

        H = int(final.shape[1])
        W = int(final.shape[2])

        if target_resolution > 0:
            sheet_longest = max(W * grid_cols, H * grid_rows)
            if sheet_longest > 0 and sheet_longest != target_resolution:
                scale = target_resolution / sheet_longest
                new_H = max(1, int(round(H * scale)))
                new_W = max(1, int(round(W * scale)))
                final_chw = final.permute(0, 3, 1, 2)
                final_chw = F.interpolate(final_chw, size=(new_H, new_W),
                                          mode="bilinear", align_corners=False)
                final = final_chw.permute(0, 2, 3, 1).contiguous()
                if input_alpha is not None:
                    in_a = F.interpolate(input_alpha.unsqueeze(1), size=(new_H, new_W),
                                         mode="bilinear", align_corners=False)
                    input_alpha = in_a.squeeze(1).clamp(0.0, 1.0).contiguous()
                H, W = new_H, new_W

        cells = grid_rows * grid_cols
        n_use = min(n_final, cells)

        sheet_h = H * grid_rows
        sheet_w = W * grid_cols

        sheet = torch.zeros((1, sheet_h, sheet_w, 4), dtype=torch.float32)
        mask_sheet = torch.zeros((1, sheet_h, sheet_w), dtype=torch.float32)

        if bg_removal == "per-frame":
            per_frame_alpha = predict_mask(final[:n_use], model, batch_size=batch_size)
            if input_alpha is not None:
                per_frame_alpha = per_frame_alpha * input_alpha[:n_use]
            padded_frames = []
            padded_alphas = []
            for i in range(n_use):
                pf, pa = _pad_frame(
                    final[i], per_frame_alpha[i],
                    padding_top, padding_bottom, padding_left, padding_right,
                    position, crop_padding, fit_to_canvas, original_image_scale,
                )
                padded_frames.append(pf)
                padded_alphas.append(pa)
            final = torch.stack(padded_frames, dim=0)
            per_frame_alpha = torch.stack(padded_alphas, dim=0)
        elif bg_removal == "none" and input_alpha is not None:
            anchored_frames = []
            anchored_alphas = []
            for i in range(n_use):
                pf, pa = _pad_frame(
                    final[i], input_alpha[i],
                    padding_top, padding_bottom, padding_left, padding_right,
                    position, crop_padding, fit_to_canvas, original_image_scale,
                )
                anchored_frames.append(pf)
                anchored_alphas.append(pa)
            final = torch.stack(anchored_frames, dim=0)
            input_alpha = torch.stack(anchored_alphas, dim=0)
            per_frame_alpha = None
        else:
            per_frame_alpha = None

        for i in range(n_use):
            row = i // grid_cols
            col = i % grid_cols
            y0, y1 = row * H, (row + 1) * H
            x0, x1 = col * W, (col + 1) * W
            sheet[0, y0:y1, x0:x1, 0:3] = final[i]
            if per_frame_alpha is not None:
                a = per_frame_alpha[i]
                sheet[0, y0:y1, x0:x1, 3] = a
                mask_sheet[0, y0:y1, x0:x1] = a
            elif input_alpha is not None:
                a = input_alpha[i]
                sheet[0, y0:y1, x0:x1, 3] = a
                mask_sheet[0, y0:y1, x0:x1] = a
            else:
                sheet[0, y0:y1, x0:x1, 3] = 1.0
                mask_sheet[0, y0:y1, x0:x1] = 1.0

        if bg_removal == "whole-sheet":
            sheet_rgb = sheet[..., 0:3]
            whole_mask = predict_mask(sheet_rgb, model)
            if input_alpha is not None:
                # mask_sheet currently holds the tiled incoming alpha.
                whole_mask = whole_mask * mask_sheet
            sheet[..., 3] = whole_mask
            mask_sheet = whole_mask

        return (sheet, mask_sheet)


NODE_CLASS_MAPPINGS = {"SpriteSheetGenerator": SpriteSheetGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"SpriteSheetGenerator": "Whisker: Sprite Sheet Generator"}
