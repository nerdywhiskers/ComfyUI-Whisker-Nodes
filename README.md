# Whisker Nodes for ComfyUI

A pack of image and mask compositing utilities for ComfyUI. All nodes appear under **Add Node → whisker-nodes**.

## Installation

### Portable ComfyUI (Windows)

From the root of `ComfyUI_windows_portable`:

```bash
cd ComfyUI\custom_nodes
git clone https://github.com/jgbyte/ComfyUI-RandomCube.git
cd ..\..
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-RandomCube\requirements.txt
```

Then restart ComfyUI.

### System Python

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jgbyte/ComfyUI-RandomCube.git
pip install -r ComfyUI-RandomCube/requirements.txt
```

### Updating

```bash
cd ComfyUI/custom_nodes/ComfyUI-RandomCube
git pull
```

Restart ComfyUI after pulling.

## Nodes

### Whisker: BG Remove + Compose

Removes the background of an input image with **BiRefNet** (MIT) or **RMBG-2.0** (BRIA, non-commercial) and composes the asset onto a canvas at user-specified dimensions.

- `background`: `alpha` (transparent canvas) or `color` (solid `bg_color` canvas, alpha-blended).
- `position`: 9-grid (`top-left` … `bottom-right`).
- `resize_to_fit`: when ON, asset scales to fit the canvas minus paddings, preserving aspect.
- `scale`: multiplier on the asset's native size when `resize_to_fit` is OFF.
- `padding_top/bottom/left/right`: per-side margins; affect both fit-scaling and 9-grid placement.
- Output: 4-channel RGBA `IMAGE` + `MASK`.

The model is moved to GPU only during inference and back to CPU between calls (via `comfy.model_management`), so it can share VRAM with diffusion models.

### Whisker: Sprite Sheet Generator

Concatenates frames from an `IMAGE` batch (e.g. from VideoHelperSuite's "Load Video" or stock animated-WebP loader) into a single sprite sheet.

- `target_frame_count`: prune by step-skipping every `total // target` frames.
- `start_index` / `end_index`: applied to the **pruned** set; `-1` = last.
- `grid_cols × grid_rows`: explicit grid layout.
- `target_resolution`: longest side of the final sheet (frames are resized first, so memory tracks the output size).
- `bg_removal`: `none`, `per-frame` (BiRefNet/RMBG on each frame), or `whole-sheet` (single pass on the assembled sheet).
- `padding_top/bottom/left/right`: only effective with `per-frame` bg removal — each frame's asset is bbox-cropped and centered within `(cell − paddings)`.
- Output: 4-channel RGBA `IMAGE` + `MASK`.

### Whisker: Ratio Mask

Generates a white rectangle of a given aspect ratio on a colored canvas, fit-resized to the canvas minus paddings.

- `ratio`: free-text like `1:1`, `16:9`, `2.35:1`. Invalid input falls back to `1:1`.
- `position`: 9-grid placement within the padded inner area.
- `bg_color`: hex string (default `#000000` for the classic white-on-black look).
- `corner_radius`: in pixels, capped at half the shorter side.
- `blur`: Gaussian blur applied to the mask; the IMAGE is derived by alpha-blending white over `bg_color` through the mask, so soft edges show consistently in both outputs.
- `padding_top/bottom/left/right`: per-side margins.
- Output: 3-channel `IMAGE` + `MASK`.

### Whisker: Strip Mask Generator

Generates 2–8 strip masks that tile a container without gaps or overlap.

- `num_masks`: 2–8.
- `orientation`: `horizontal` (full-width bands) or `vertical` (full-height bands).
- `primary_mask_size` + `primary_mask_index`: one strip's pixel size is user-set; the others split the remainder evenly with the rounding remainder distributed per-pixel so the strips exactly fill the container.
- `noise_amount` / `noise_scale`: smooth 1D noise displaces the interior boundaries. All boundaries share the same noise vector, so the strip stack warps coherently.
- `blur`: Gaussian blur on each output mask.
- Output: 8 fixed `MASK` outputs (`mask_1` … `mask_8`); slots beyond `num_masks` are zero masks.

### Whisker: Random Cube Grid Generator

Procedurally places white squares on a black grid with control over density, gaps, sizing, and per-edge column rules. Outputs the image plus a JSON list of `{x, y, size}` cube coordinates for downstream use (the JSON pairs naturally with the `Block Grid Generator`-style nodes you might build on top of it).

### Whisker: Offset Image

Wraps an image by `offset_x` / `offset_y` pixels via `torch.roll` (true tile-style shift) and emits a seam mask centered on the actual wrap line, with adjustable thickness and Gaussian blur — handy for inpainting seams to make seamless tiles.

## Models

The BG removal models download to your HuggingFace cache on first use (`~/.cache/huggingface/hub/`):

- **BiRefNet** (`ZhengPeng7/BiRefNet`) — MIT license, ~880 MB. Default. Good edge quality on hair, fur, fine details.
- **RMBG-2.0** (`briaai/RMBG-2.0`) — BRIA license, **non-commercial without a paid license**, ~885 MB.

If you intend to use the BG removal nodes commercially, stick with **BiRefNet**.

## License

See [LICENSE](LICENSE).
