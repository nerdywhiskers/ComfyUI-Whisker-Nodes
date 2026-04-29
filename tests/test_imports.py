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
        "ratio_mask",
        "sprite_sheet",
    ]

    for node_name in expected_nodes:
        assert node_name in NODE_CLASS_MAPPINGS, f"Node {node_name} not found in mappings"

    assert len(NODE_CLASS_MAPPINGS) == len(expected_nodes)
    assert len(NODE_DISPLAY_NAME_MAPPINGS) == len(expected_nodes)


def test_node_categories():
    """Verify nodes have correct category."""
    from nodes.random_cube_grid import RandomCubeGrid
    from nodes.offset_image import OffsetImageNode
    from nodes.bg_remove import BGRemoveCompose

    assert RandomCubeGrid.CATEGORY == "whisker-nodes"
    assert OffsetImageNode.CATEGORY == "whisker-nodes"
    assert BGRemoveCompose.CATEGORY == "whisker-nodes"
