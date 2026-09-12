"""Test that all node modules can be imported successfully."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_all_nodes():
    """Verify all node classes can be imported."""
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    expected_nodes = [
        "RandomCubeGrid",
        "offset_image",
        "bg_remove_compose",
        "strip_masks",
        "shape_mask",
        # Legacy id for workflows saved with the old Ratio Mask node.
        "ratio_mask",
        "sprite_sheet",
    ]

    for node_name in expected_nodes:
        assert node_name in NODE_CLASS_MAPPINGS, f"Node {node_name} not found in mappings"

    assert len(NODE_CLASS_MAPPINGS) == len(expected_nodes)
    # The legacy ratio_mask alias has no display entry (frontend falls back
    # to the id), so display mappings hold one entry per visible node.
    assert len(NODE_DISPLAY_NAME_MAPPINGS) == len(expected_nodes) - 1
    assert NODE_DISPLAY_NAME_MAPPINGS["shape_mask"] == "Whisker: Shape Mask"


def test_node_categories():
    """Verify nodes have correct category."""
    from nodes.random_cube_grid import RandomCubeGrid
    from nodes.offset_image import OffsetImageNode
    from nodes.bg_remove import BGRemoveCompose
    from nodes.shape_mask import ShapeMask

    assert RandomCubeGrid.CATEGORY == "whisker-nodes"
    assert OffsetImageNode.CATEGORY == "whisker-nodes"
    assert BGRemoveCompose.CATEGORY == "whisker-nodes"
    assert ShapeMask.CATEGORY == "whisker-nodes"
