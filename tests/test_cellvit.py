"""Tests for the CellViT detection backend wrappers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_detect_cells_core_returns_documented_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend should return the documented tuple of cell dictionaries."""

    # Provide a minimal OpenCV stub before importing the module under test.
    fake_cv2 = types.SimpleNamespace(
        BORDER_CONSTANT=0,
        copyMakeBorder=lambda img, top, bottom, left, right, border_type, value=None: img,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    fake_tqdm_module = types.ModuleType("tqdm")
    fake_tqdm_module.tqdm = lambda iterable, total=None: iterable
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm_module)

    class FakeNDArray(list):
        def tolist(self):
            return list(self)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda values: FakeNDArray(list(values))

    def fake_ceil(value):
        from math import ceil

        return ceil(value)

    fake_numpy.ceil = fake_ceil
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    class FakeIndex(list):
        @property
        def values(self):
            return list(self)

    class FakeDataFrame:
        def __init__(self, index=None):
            self.index = FakeIndex([] if index is None else list(index))

    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = FakeDataFrame
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    class FakeImage:
        def __init__(self, size, mode="RGB"):
            self.size = tuple(size)
            self.mode = mode

        def crop(self, box):
            x0, y0, x1, y1 = box
            width = max(0, x1 - x0)
            height = max(0, y1 - y0)
            return FakeImage((width, height), self.mode)

        def paste(self, image, box):
            # The real implementation would copy pixels; the size is already correct.
            return None

    fake_image_module = types.ModuleType("PIL.Image")
    fake_image_module.Image = FakeImage
    fake_image_module.MAX_IMAGE_PIXELS = None

    def fake_new(mode, size, color=(255, 255, 255)):
        return FakeImage(size, mode)

    fake_image_module.new = fake_new
    fake_pil_pkg = types.ModuleType("PIL")
    fake_pil_pkg.Image = fake_image_module
    monkeypatch.setitem(sys.modules, "PIL", fake_pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    fake_panospace_pkg = types.ModuleType("panospace")
    fake_panospace_pkg.__path__ = [str(PROJECT_ROOT / "panospace")]
    monkeypatch.setitem(sys.modules, "panospace", fake_panospace_pkg)

    from panospace._core.detection import cellvit
    from PIL import Image  # type: ignore

    # Replace the heavy detector with a lightweight stub.
    class DummyDetector:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def detect_patch(self, patch):  # pragma: no cover - trivial stub
            return object(), [object()]

    monkeypatch.setattr(cellvit, "_lazy_detector", lambda: DummyDetector)

    sample_bbox = [0, 0, 5, 5]
    sample_centroid = [2, 3]
    sample_contour = [[0, 0], [1, 1], [2, 2]]
    sample_type = "tumour"

    def fake_process_cell_instance(
        *,
        instance_types,
        offset_global,
        row,
        col,
        tile_size,
        overlap,
    ):
        cell_dict = {
            "bbox": sample_bbox,
            "centroid": sample_centroid,
            "contour": sample_contour,
            "type": sample_type,
            "patch_coordinates": [row, col],
            "cell_status": 0,
            "offset_global": offset_global.tolist(),
        }
        detection_dict = {
            "bbox": sample_bbox,
            "centroid": sample_centroid,
            "type": sample_type,
        }
        return [cell_dict], [detection_dict]

    class FakeCellPostProcessor:
        def __init__(self, cell_list):
            self.cell_list = cell_list

        def post_process_cells(self):
            return FakeDataFrame(index=range(len(self.cell_list)))

    fake_backend_pkg = types.ModuleType("panospace._core.detection._cellvit_backend")
    fake_backend_pkg.__path__ = []  # mark as package
    fake_postprocessing_module = types.ModuleType(
        "panospace._core.detection._cellvit_backend.postprocessing"
    )
    fake_postprocessing_module.process_cell_instance = fake_process_cell_instance
    fake_postprocessing_module.CellPostProcessor = FakeCellPostProcessor
    fake_backend_pkg.postprocessing = fake_postprocessing_module

    monkeypatch.setitem(
        sys.modules,
        "panospace._core.detection._cellvit_backend",
        fake_backend_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "panospace._core.detection._cellvit_backend.postprocessing",
        fake_postprocessing_module,
    )

    img = Image.new("RGB", (64, 64), color=0)

    cells, detections = cellvit.detect_cells_core(
        img, model_name="HIPT", device="cpu", tile_size=64, overlap=0
    )

    assert isinstance(cells, list)
    assert isinstance(detections, list)
    assert cells and detections
    assert all(isinstance(cell, dict) for cell in cells)
    assert {
        "bbox",
        "centroid",
        "contour",
        "type",
        "patch_coordinates",
        "cell_status",
        "offset_global",
    }.issubset(cells[0].keys())
    assert cells[0]["bbox"] == sample_bbox
    assert cells[0]["centroid"] == sample_centroid
    assert cells[0]["type"] == sample_type
    assert cells[0]["patch_coordinates"] == [1, 1]
    assert cells[0]["offset_global"] == [0, 0]

    assert all(isinstance(det, dict) for det in detections)
    assert detections[0] == {
        "bbox": sample_bbox,
        "centroid": sample_centroid,
        "type": sample_type,
    }
