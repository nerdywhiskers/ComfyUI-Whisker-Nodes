import json
import numpy as np
import torch
import random

class RandomCubeGrid:
    """
    Generates a grid with randomly placed cubes.

    Inputs:
    - Grid width (units)
    - Grid height (units)
    - Output resolution (pixels) - longest side
    - Minimum cube size (units)
    - Maximum cube size (units)
    - First column start row
    - First column end row
    - Last column start row
    - Last column end row
    - Minimum column gap (randomly applied between cubes in a row)
    - Maximum column gap (upper limit for column gap randomness)
    - Minimum row gap (randomly applied between cubes in a column)
    - Maximum row gap (upper limit for row gap randomness)
    - Density (probability of placing a cube in each available spot)
    - Apply density & gaps to first & last column? (toggle)
    - Seed (for randomization, -1 for random seed)

    Outputs:
    - PNG image
    - JSON string: list of {"x", "y", "size"} objects for cube coordinates
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_width": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "grid_height": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "output_resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "min_cube_size": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "max_cube_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "first_col_start_row": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1}),
                "first_col_end_row": ("INT", {"default": 7, "min": 0, "max": 100, "step": 1}),
                "last_col_start_row": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "last_col_end_row": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "min_col_gap": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                "max_col_gap": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "min_row_gap": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                "max_row_gap": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "density": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "first_last_density_toggle": (["enabled", "disabled"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("cube_image", "cube_coordinates")

    FUNCTION = "generate_grid"

    CATEGORY = "whisker-nodes"

    def generate_grid(
        self, grid_width, grid_height, output_resolution, min_cube_size, max_cube_size,
        first_col_start_row, first_col_end_row, last_col_start_row, last_col_end_row,
        min_col_gap, max_col_gap, min_row_gap, max_row_gap, density, first_last_density_toggle, seed
    ):
        # Set random seed
        if seed == -1:
            seed = random.randint(0, 999999)
        random.seed(seed)

        # Ensure valid row ranges
        if first_col_start_row > first_col_end_row:
            first_col_start_row, first_col_end_row = first_col_end_row, first_col_start_row
        if last_col_start_row > last_col_end_row:
            last_col_start_row, last_col_end_row = last_col_end_row, last_col_start_row

        # Determine aspect ratio and resolution scaling
        aspect_ratio = grid_width / grid_height
        if grid_width > grid_height:
            width_px = output_resolution
            height_px = int(output_resolution / aspect_ratio)
        else:
            height_px = output_resolution
            width_px = int(output_resolution * aspect_ratio)

        # Create blank black image
        image = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        # Compute unit size
        unit_width = width_px / grid_width
        unit_height = height_px / grid_height

        # Store cube coordinates
        cube_coordinates = []

        col_positions = list(range(0, grid_width))

        # Apply column gaps unless first/last density is disabled
        selected_cols = []
        last_col = -999
        for col in col_positions:
            if first_last_density_toggle == "disabled" and (col == 0 or col == grid_width - 1):
                selected_cols.append(col)  # Ensure first & last column are always placed
            elif col - last_col >= random.randint(min_col_gap, max_col_gap):
                selected_cols.append(col)
                last_col = col

        for col in selected_cols:
            col_px = int(col * unit_width)

            # Determine row range
            if col == 0:
                row_range = range(first_col_start_row, min(first_col_end_row + 1, grid_height))
            elif col == grid_width - 1:
                row_range = range(last_col_start_row, min(last_col_end_row + 1, grid_height))
            else:
                row_range = range(0, grid_height)

            # Apply density only if enabled for first/last columns
            if first_last_density_toggle == "enabled" or (col != 0 and col != grid_width - 1):
                row_candidates = [row for row in row_range if random.random() < density]
            else:
                row_candidates = list(row_range)  # Force placement if disabled

            # Apply row gaps unless first/last density is disabled
            selected_rows = []
            last_row = -999
            for row in row_candidates:
                if first_last_density_toggle == "disabled" and (col == 0 or col == grid_width - 1):
                    selected_rows.append(row)  # Ensure full row placement
                elif row - last_row >= random.randint(min_row_gap, max_row_gap):
                    selected_rows.append(row)
                    last_row = row

            for row in selected_rows:
                row_px = int(row * unit_height)

                # Random cube size in unit scale
                cube_size_units = random.uniform(min_cube_size, max_cube_size)
                cube_width_px = int(cube_size_units * unit_width)
                cube_height_px = int(cube_size_units * unit_height)

                # Ensure the cube fits in the output resolution
                if col_px + cube_width_px < width_px and row_px + cube_height_px < height_px:
                    image[row_px:row_px+cube_height_px, col_px:col_px+cube_width_px] = [255, 255, 255]
                    cube_coordinates.append((col, row, cube_size_units))

        # Convert image to torch tensor in (B, H, W, C) format
        image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        coordinates_json = json.dumps(
            [{"x": x, "y": y, "size": round(float(size), 2)} for x, y, size in cube_coordinates]
        )

        return (image_tensor, coordinates_json)

# Register node
NODE_CLASS_MAPPINGS = {"RandomCubeGrid": RandomCubeGrid}
NODE_DISPLAY_NAME_MAPPINGS = {"RandomCubeGrid": "Whisker: Random Cube Grid Generator"}
