import torch
import torch.nn.functional as F

from utils.bg_remove_utils import MODEL_REGISTRY, predict_mask, mask_bbox


def _pad_frame(frame, alpha, pad_top, pad_bottom, pad_left, pad_right):
    """
    Bbox the asset using alpha, scale-to-fit (no upscale) within
    (cell - paddings) preserving aspect ratio, and center inside the padded
    area. Returns a frame and alpha of the same shape as the input frame.
    """
    H, W = int(frame.shape[0]), int(frame.shape[1])
    bbox = mask_bbox(alpha)
    if bbox is None:
        return torch.zeros_like(frame), torch.zeros_like(alpha)

    y0, x0, y1, x1 = bbox
    asset = frame[y0:y1, x0:x1, :]
    asset_a = alpha[y0:y1, x0:x1]
    ah, aw = int(asset.shape[0]), int(asset.shape[1])

    avail_h = max(1, H - pad_top - pad_bottom)
    avail_w = max(1, W - pad_left - pad_right)

    fit_scale = min(avail_h / ah, avail_w / aw, 1.0)
    new_h = max(1, int(round(ah * fit_scale)))
    new_w = max(1, int(round(aw * fit_scale)))

    if new_h != ah or new_w != aw:
        asset = F.interpolate(
            asset.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).contiguous()
        asset_a = F.interpolate(
            asset_a.unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    cy = pad_top + (avail_h - new_h) // 2
    cx = pad_left + (avail_w - new_w) // 2

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

    padding_top/bottom/left/right are only effective when bg_removal is
    'per-frame'. Each frame's asset is bbox-cropped, scaled to fit within
    (cell - paddings) preserving aspect ratio (never upscaled), and
    centered within the padded area. Ignored otherwise.

    The output IMAGE is always 4-channel RGBA. The MASK output mirrors the
    sheet's alpha channel (all 1s when bg_removal is 'none').

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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "whisker-nodes"

    def generate(self, frames, target_frame_count, start_index, end_index,
                 grid_cols, grid_rows, target_resolution, bg_removal, model,
                 padding_top, padding_bottom, padding_left, padding_right):
        final = _prune_frames(frames, target_frame_count, start_index, end_index)
        n_final = int(final.shape[0])

        if n_final == 0:
            empty_img = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return (empty_img, empty_mask)

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
                H, W = new_H, new_W

        cells = grid_rows * grid_cols
        n_use = min(n_final, cells)

        sheet_h = H * grid_rows
        sheet_w = W * grid_cols

        sheet = torch.zeros((1, sheet_h, sheet_w, 4), dtype=torch.float32)
        mask_sheet = torch.zeros((1, sheet_h, sheet_w), dtype=torch.float32)

        if bg_removal == "per-frame":
            per_frame_alpha = predict_mask(final[:n_use], model)
            if padding_top or padding_bottom or padding_left or padding_right:
                padded_frames = []
                padded_alphas = []
                for i in range(n_use):
                    pf, pa = _pad_frame(
                        final[i], per_frame_alpha[i],
                        padding_top, padding_bottom, padding_left, padding_right,
                    )
                    padded_frames.append(pf)
                    padded_alphas.append(pa)
                final = torch.stack(padded_frames, dim=0)
                per_frame_alpha = torch.stack(padded_alphas, dim=0)
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
            else:
                sheet[0, y0:y1, x0:x1, 3] = 1.0
                mask_sheet[0, y0:y1, x0:x1] = 1.0

        if bg_removal == "whole-sheet":
            sheet_rgb = sheet[..., 0:3]
            whole_mask = predict_mask(sheet_rgb, model)
            sheet[..., 3] = whole_mask
            mask_sheet = whole_mask

        return (sheet, mask_sheet)


NODE_CLASS_MAPPINGS = {"SpriteSheetGenerator": SpriteSheetGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"SpriteSheetGenerator": "Whisker: Sprite Sheet Generator"}
