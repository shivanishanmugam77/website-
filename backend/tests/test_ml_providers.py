"""The parts of the model wrappers that do not need a model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from app.ml.detection import get_detection_provider
from app.ml.detection.grounding_dino import GroundingDinoProvider, pick_device, shrink
from app.ml.errors import ModelUnavailableError
from app.ml.segmentation import get_segmentation_provider


def _torch_is_installed() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def test_a_big_picture_is_shrunk_and_the_factor_to_undo_it_is_reported() -> None:
    small, factor = shrink(Image.new("RGB", (2048, 1024)), 1024)
    assert small.size == (1024, 512) and factor == 2.0
    unchanged, one = shrink(Image.new("RGB", (800, 600)), 1024)
    assert unchanged.size == (800, 600) and one == 1.0


def test_a_tall_picture_is_shrunk_by_its_longest_side() -> None:
    small, factor = shrink(Image.new("RGB", (500, 3000)), 1000)
    assert small.size == (167, 1000) and factor == 3.0


@pytest.mark.parametrize(
    ("wanted", "cuda_available", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cpu", True, "cpu"), ("cuda", False, "cuda")],
)
def test_the_device_follows_the_setting_and_what_exists(
    wanted: str, cuda_available: bool, expected: str
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))
    assert pick_device(fake_torch, wanted) == expected


def test_unknown_providers_are_reported_clearly() -> None:
    with pytest.raises(ModelUnavailableError, match="Unknown detection provider"):
        get_detection_provider(
            SimpleNamespace(
                detection_provider="nonexistent",
                detection_model="a/b",
                detection_box_threshold=0.3,
                detection_text_threshold=0.2,
                detection_max_px=1024,
                ml_num_threads=1,
                ml_device="cpu",
            )
        )
    with pytest.raises(ModelUnavailableError, match="Unknown segmentation provider"):
        get_segmentation_provider(
            SimpleNamespace(
                segmentation_provider="nonexistent",
                segmentation_model="a/b",
                ml_num_threads=1,
                ml_device="cpu",
            )
        )


@pytest.mark.skipif(_torch_is_installed(), reason="only meaningful where torch is missing")
def test_a_missing_ml_package_is_explained_not_a_crash() -> None:
    with pytest.raises(ModelUnavailableError, match="not installed"):
        GroundingDinoProvider(
            "IDEA-Research/grounding-dino-tiny",
            box_threshold=0.3,
            text_threshold=0.2,
            max_px=1024,
            threads=1,
            device="cpu",
        )
