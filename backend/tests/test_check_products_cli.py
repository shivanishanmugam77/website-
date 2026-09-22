"""The ``check_products`` command, with stand-in models."""

from __future__ import annotations

import fnmatch
import sys
from types import ModuleType

import pytest
from PIL import Image
from sqlalchemy import func, select

from app.cli import check_products, download_models
from app.ml.detection.base import Detection
from app.ml.errors import ModelUnavailableError
from app.ml.segmentation.base import Segment
from app.models import SourceImage
from tests.factories import make_catalogue, make_page
from tests.test_panel_editing_api import editable_page  # a 1200 x 800 page with two panels


class FakeFinder:
    name = "fake-finder"
    version = "fake 1"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], list[str]]] = []

    def detect(self, image: Image.Image, prompts):  # noqa: ANN001, ANN201
        self.calls.append((image.size, list(prompts)))
        if image.size == (500, 600):  # the big panel of the test page
            return [
                Detection("desk", 0.82, (50, 250, 450, 500)),
                Detection("office chair", 0.71, (60, 60, 220, 300)),
            ]
        return []


class FakeOutliner:
    name = "fake-outliner"
    version = "fake 1"

    def __init__(self) -> None:
        self.boxes: list[list[tuple]] = []

    def segment(self, image: Image.Image, boxes):  # noqa: ANN001, ANN201
        self.boxes.append(list(boxes))
        return [Segment(mask=Image.new("L", image.size, 255), score=0.9) for _ in boxes]


def run(catalogue, tmp_path, *extra, page: int = 1, **providers):  # noqa: ANN001, ANN002, ANN003, ANN201
    return check_products.main(
        ["--catalogue", str(catalogue.id), "--page", str(page), "--out", str(tmp_path / "out.jpg"), *extra],
        **providers,
    )


def test_it_reports_what_was_found_on_every_panel(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    finder = FakeFinder()
    assert run(catalogue, tmp_path, session=db_session, storage=storage, detector=finder) == 0

    output = capsys.readouterr().out
    assert "Panel 1 (500 x 600): 2 found" in output and "Panel 2 (400 x 300): 0 found" in output
    assert "desk" in output and "82%" in output and "office chair" in output and "71%" in output
    assert "2 things found in all." in output
    assert [size for size, _ in finder.calls] == [(500, 600), (400, 300)]
    assert "desk" in finder.calls[0][1] and "office chair" in finder.calls[0][1]  # what it was asked

    picture = Image.open(tmp_path / "out.jpg")
    assert picture.format == "JPEG" and picture.width >= 500


def test_one_panel_can_be_tried_alone(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    finder = FakeFinder()
    assert run(catalogue, tmp_path, "--panel", "2", session=db_session, storage=storage, detector=finder) == 0
    assert [size for size, _ in finder.calls] == [(400, 300)]
    assert "Panel 1" not in capsys.readouterr().out


def test_outlines_are_only_made_when_asked_for(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    without = FakeOutliner()
    run(catalogue, tmp_path, session=db_session, storage=storage, detector=FakeFinder(), segmenter=without)
    assert without.boxes == []
    assert "outline fills" not in capsys.readouterr().out

    outliner = FakeOutliner()
    code = run(
        catalogue, tmp_path, "--masks", session=db_session, storage=storage, detector=FakeFinder(), segmenter=outliner
    )
    assert code == 0 and len(outliner.boxes) == 1  # only the panel with finds is outlined
    assert outliner.boxes[0] == [(60, 60, 220, 300), (50, 250, 450, 500)]  # in reading order
    assert capsys.readouterr().out.count("outline fills 100% of its box") == 2  # a full-size stand-in outline


def test_nothing_is_saved(db_session, storage, tmp_path) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    before = db_session.scalar(select(func.count()).select_from(SourceImage))
    run(catalogue, tmp_path, session=db_session, storage=storage, detector=FakeFinder())
    assert db_session.scalar(select(func.count()).select_from(SourceImage)) == before


def test_unknown_pages_and_panels_are_reported(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    finder = FakeFinder()
    assert run(catalogue, tmp_path, page=9, session=db_session, storage=storage, detector=finder) == 1
    assert "No page 9" in capsys.readouterr().err
    assert run(catalogue, tmp_path, "--panel", "9", session=db_session, storage=storage, detector=finder) == 1
    assert "No such panel" in capsys.readouterr().err
    assert finder.calls == []


def test_a_page_without_panels_is_reported(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    code = run(catalogue, tmp_path, session=db_session, storage=storage, detector=FakeFinder())
    assert code == 1 and "no photo panels" in capsys.readouterr().out


def test_a_model_that_cannot_load_is_explained(db_session, storage, tmp_path, monkeypatch, capsys) -> None:  # noqa: ANN001
    def broken(_settings):  # noqa: ANN001, ANN202
        raise ModelUnavailableError("the model is not downloaded yet")

    monkeypatch.setattr(check_products, "get_detection_provider", broken)
    catalogue, *_ = editable_page(db_session, storage)
    assert run(catalogue, tmp_path, session=db_session, storage=storage) == 1
    assert "FAILED: the model is not downloaded yet" in capsys.readouterr().err


def test_an_unwritable_output_place_does_not_lose_the_results(db_session, storage, tmp_path, capsys) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    blocker = tmp_path / "a-file"
    blocker.write_text("in the way")
    code = check_products.main(
        ["--catalogue", str(catalogue.id), "--page", "1", "--out", str(blocker / "x.jpg")],
        session=db_session,
        storage=storage,
        detector=FakeFinder(),
    )
    captured = capsys.readouterr()
    assert code == 0 and "Could not write the picture" in captured.err and "desk" in captured.out


# ------------------------------------------------------------------------------ download_models
class FakeHub(ModuleType):
    """A stand-in for huggingface_hub with one repository that only has ``.bin`` weights."""

    def __init__(self, folder, files) -> None:  # noqa: ANN001
        super().__init__("huggingface_hub")
        self.folder, self.files, self.asked = folder, files, []

    def snapshot_download(self, repo_id, allow_patterns):  # noqa: ANN001, ANN201
        self.asked.append((repo_id, list(allow_patterns)))
        for name in self.files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                (self.folder / name).write_bytes(b"x" * 1000)
        return str(self.folder)


def test_weights_are_fetched_in_one_format_falling_back_when_missing(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    hub = FakeHub(tmp_path, ["config.json", "model.bin", "model.h5"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    folder = download_models._download("a/b")
    assert folder == tmp_path
    assert (tmp_path / "model.bin").exists() and not (tmp_path / "model.h5").exists()
    assert "*.safetensors" in hub.asked[0][1] and "*.bin" in hub.asked[1][1]


def test_safetensors_are_preferred_when_present(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    hub = FakeHub(tmp_path, ["config.json", "model.safetensors", "model.bin"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    download_models._download("a/b")
    assert len(hub.asked) == 1 and not (tmp_path / "model.bin").exists()


def test_the_download_command_reports_success_and_failure(tmp_path, monkeypatch, capsys) -> None:  # noqa: ANN001
    (tmp_path / "w.safetensors").write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(download_models, "_download", lambda repo: tmp_path)
    assert download_models.main([]) == 0
    assert "All models are downloaded." in capsys.readouterr().out

    def failing(repo):  # noqa: ANN001, ANN202
        raise OSError("no internet")

    monkeypatch.setattr(download_models, "_download", failing)
    assert download_models.main([]) == 1
    assert "FAILED: OSError: no internet" in capsys.readouterr().err


def test_a_repository_without_usable_weights_is_an_error(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub(tmp_path, ["config.json"]))
    with pytest.raises(RuntimeError, match="no weights"):
        download_models._download("a/b")
