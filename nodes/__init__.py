from .random_cube_grid import NODE_CLASS_MAPPINGS as RANDOM_GRID_NODES, NODE_DISPLAY_NAME_MAPPINGS as RANDOM_GRID_DISPLAY_NAMES
from .offset_image import OffsetImageNode
from .bg_remove import BGRemoveCompose
from .strip_masks import StripMaskGenerator
from .shape_mask import ShapeMask
from .sprite_sheet import SpriteSheetGenerator

NODE_CLASS_MAPPINGS = {
    **RANDOM_GRID_NODES,
    "offset_image": OffsetImageNode,
    "bg_remove_compose": BGRemoveCompose,
    "strip_masks": StripMaskGenerator,
    "shape_mask": ShapeMask,
    # Legacy id for workflows saved with the old Ratio Mask node.
    "ratio_mask": ShapeMask,
    "sprite_sheet": SpriteSheetGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RANDOM_GRID_DISPLAY_NAMES,
    "offset_image": "Whisker: Offset Image",
    "bg_remove_compose": "Whisker: BG Remove + Compose",
    "strip_masks": "Whisker: Strip Mask Generator",
    "shape_mask": "Whisker: Shape Mask",
    "sprite_sheet": "Whisker: Sprite Sheet Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
