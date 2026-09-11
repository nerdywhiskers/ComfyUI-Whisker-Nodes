"""Behavior tests for BG Remove + Compose using synthetic model masks."""

import pytest

torch = pytest.importorskip("torch")
from nodes import bg_remove  # noqa: E402

EXPECTED_INPUTS = [
    "image",
    "model",
    "width",
    "height",
    "background",
    "bg_color",
    "position",
    "resize_to_fit",
    "scale",
    "padding_top",
    "padding_bottom",
    "padding_left",
    "padding_right",
    "crop_padding",
]


def compose(monkeypatch, image, mask, **overrides):
    monkeypatch.setattr(bg_remove, "predict_mask", lambda _image, _model: mask)
    args = dict(
        model="BiRefNet",
        width=20,
        height=20,
        background="alpha",
        bg_color="#ffffff",
        position="middle-center",
        resize_to_fit=False,
        scale=1.0,
        padding_top=0,
        padding_bottom=0,
        padding_left=0,
        padding_right=0,
        crop_padding=0,
    )
    args.update(overrides)
    return bg_remove.BGRemoveCompose().compose(image, **args)


def nonzero_bounds(mask):
    points = (mask > 0.05).nonzero()
    if not len(points):
        return None
    return tuple(points.min(0).values.tolist() + (points.max(0).values + 1).tolist())


def test_input_schema_preserves_order_and_appends_crop_padding():
    required = bg_remove.BGRemoveCompose.INPUT_TYPES()["required"]
    assert list(required) == EXPECTED_INPUTS
    assert required["crop_padding"][1]["default"] == 20


@pytest.mark.parametrize("padding,bounds", [(0, (0, 0, 5, 5)), (20, (1, 2, 6, 7))])
def test_crop_padding_and_source_edge_clipping(monkeypatch, padding, bounds):
    image = torch.ones((1, 8, 9, 3))
    mask = torch.zeros((1, 8, 9))
    mask[:, 1:6, 2:7] = 1
    _, output_mask = compose(
        monkeypatch,
        image,
        mask,
        width=9,
        height=8,
        position="top-left",
        crop_padding=padding,
    )
    assert nonzero_bounds(output_mask[0]) == bounds


@pytest.mark.parametrize("position", bg_remove.POSITIONS)
@pytest.mark.parametrize("pads", [(0, 0, 0, 0), (2, 3, 4, 5)])
def test_all_canvas_anchors(monkeypatch, position, pads):
    top, bottom, left, right = pads
    image = torch.ones((1, 2, 3, 3))
    mask = torch.ones((1, 2, 3))
    _, output_mask = compose(
        monkeypatch,
        image,
        mask,
        width=20,
        height=16,
        position=position,
        padding_top=top,
        padding_bottom=bottom,
        padding_left=left,
        padding_right=right,
    )
    y, x = bg_remove.resolve_canvas_anchor(position, 16, 20, 2, 3, *pads)
    assert nonzero_bounds(output_mask[0]) == (y, x, y + 2, x + 3)


@pytest.mark.parametrize("resize_to_fit,scale", [(True, 1.0), (False, 10.0)])
def test_resize_is_proportional_and_does_not_clip(monkeypatch, resize_to_fit, scale):
    image = torch.ones((1, 4, 8, 3))
    mask = torch.ones((1, 4, 8))
    _, output_mask = compose(
        monkeypatch,
        image,
        mask,
        width=12,
        height=12,
        position="bottom-right",
        resize_to_fit=resize_to_fit,
        scale=scale,
        padding_top=2,
        padding_bottom=2,
        padding_left=2,
        padding_right=2,
    )
    assert nonzero_bounds(output_mask[0]) == (6, 2, 10, 10)


def test_alpha_color_batch_and_empty_mask(monkeypatch):
    image = torch.zeros((2, 2, 2, 3))
    image[0, ..., 0] = 1
    mask = torch.zeros((2, 2, 2))
    mask[0] = 0.5
    alpha_image, alpha_mask = compose(monkeypatch, image, mask, width=4, height=4)
    assert alpha_image.shape == (2, 4, 4, 4) and alpha_mask.shape == (2, 4, 4)
    assert torch.allclose(alpha_image[0, 1:3, 1:3, 3], torch.full((2, 2), 0.5))
    assert not alpha_image[1].any() and not alpha_mask[1].any()
    color_image, color_mask = compose(
        monkeypatch,
        image,
        mask,
        width=4,
        height=4,
        background="color",
        bg_color="#0000ff",
    )
    assert torch.all(color_image[..., 3] == 1)
    assert torch.allclose(color_image[0, 1, 1, :3], torch.tensor([0.5, 0.0, 0.5]))
    assert torch.all(color_image[1, ..., 0:2] == 0)
    assert torch.all(color_image[1, ..., 2] == 1)
    assert not color_mask[1].any()
