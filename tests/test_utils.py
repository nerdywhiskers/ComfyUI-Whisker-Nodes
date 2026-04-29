"""Test utility functions from utils/bg_remove_utils.py."""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hex_to_rgb_valid():
    """Test valid hex color conversion."""
    from utils.bg_remove_utils import hex_to_rgb

    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000FF") == (0, 0, 255)
    assert hex_to_rgb("FFF") == (255, 255, 255)
    assert hex_to_rgb("000") == (0, 0, 0)


def test_hex_to_rgb_invalid():
    """Test invalid hex color raises ValueError."""
    from utils.bg_remove_utils import hex_to_rgb

    import pytest
    try:
        hex_to_rgb("GGGGGG")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        hex_to_rgb("12345")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_mask_bbox():
    """Test bounding box calculation from mask."""
    from utils.bg_remove_utils import mask_bbox

    # Create a mask with a rectangle in the middle
    mask = torch.zeros(64, 64)
    mask[10:30, 20:50] = 1.0

    bbox = mask_bbox(mask)
    assert bbox is not None
    y0, x0, y1, x1 = bbox
    assert y0 == 10
    assert x0 == 20
    assert y1 == 30
    assert x1 == 50


def test_mask_bbox_empty():
    """Test empty mask returns None."""
    from utils.bg_remove_utils import mask_bbox

    mask = torch.zeros(32, 32)
    assert mask_bbox(mask) is None
