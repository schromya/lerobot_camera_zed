# Copyright 2026 Stereolabs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example of running a specific test:
# ```bash
# pytest tests/test_find_cameras.py::test_lists_stereo_and_monocular
# ```

# `lerobot-find-zed-cameras` glues discovery of two SDK classes together and must not take the
# whole listing down when one family is unavailable. Only the glue is tested here; the camera
# backends have their own suites.

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_camera_zed import ZedCamera, ZedOneCamera, find_cameras

STEREO = {"name": "ZED X", "type": "zed", "id": 1, "serial_number": 1}
MONO = {"name": "ZED X One GS", "type": "zed_one", "id": 2, "serial_number": 2}


@pytest.fixture(autouse=True)
def _no_global_logging_setup(monkeypatch):
    """`main()` calls `init_logging()`, which reconfigures the root logger for the whole session."""
    monkeypatch.setattr(find_cameras, "init_logging", lambda: None)


@pytest.fixture(name="fake_finders")
def fixture_fake_finders(monkeypatch):
    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(lambda: [STEREO]))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(lambda: [MONO]))


def test_lists_stereo_and_monocular(fake_finders):
    assert find_cameras.find_zed_cameras() == [STEREO, MONO]


def test_stereo_failure_does_not_hide_monocular(monkeypatch, caplog):
    def boom():
        raise RuntimeError("no SDK")

    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(boom))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(lambda: [MONO]))

    with caplog.at_level(logging.WARNING):
        assert find_cameras.find_zed_cameras() == [MONO]

    assert "stereo discovery failed" in caplog.text


def test_monocular_failure_does_not_hide_stereo(monkeypatch, caplog):
    def boom():
        raise RuntimeError("no SDK")

    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(lambda: [STEREO]))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(boom))

    with caplog.at_level(logging.WARNING):
        assert find_cameras.find_zed_cameras() == [STEREO]

    assert "monocular discovery failed" in caplog.text


def test_no_sdk_returns_empty_instead_of_raising(monkeypatch, caplog):
    def boom():
        raise ImportError("pyzed is not installed")

    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(boom))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(boom))

    with caplog.at_level(logging.WARNING):
        assert find_cameras.find_zed_cameras() == []


def test_main_prints_every_detected_camera(fake_finders, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["lerobot-find-zed-cameras"])

    find_cameras.main()

    out = capsys.readouterr().out
    assert "ZED X  type=zed  serial_number=1" in out
    assert "ZED X One GS  type=zed_one  serial_number=2" in out


def test_main_without_cameras_warns_and_saves_nothing(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(ZedCamera, "find_cameras", staticmethod(list))
    monkeypatch.setattr(ZedOneCamera, "find_cameras", staticmethod(list))
    monkeypatch.setattr(
        "sys.argv", ["lerobot-find-zed-cameras", "--save-images", "--output-dir", str(tmp_path / "out")]
    )

    with caplog.at_level(logging.WARNING):
        find_cameras.main()

    assert "No ZED cameras were detected" in caplog.text
    assert not (tmp_path / "out").exists()  # nothing to capture, so no directory is created


class _SpyCamera:
    """Stands in for a backend: records the config it was built with, serves one frame."""

    instances: list["_SpyCamera"] = []

    def __init__(self, config):
        self.config = config
        type(self).instances.append(self)

    def connect(self):
        pass

    def read(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def disconnect(self):
        pass


class _SpyStereo(_SpyCamera):
    instances: list["_SpyCamera"] = []


class _SpyMono(_SpyCamera):
    instances: list["_SpyCamera"] = []


@pytest.fixture(name="spies")
def fixture_spies(monkeypatch):
    _SpyStereo.instances = []
    _SpyMono.instances = []
    monkeypatch.setattr(find_cameras, "ZedCamera", _SpyStereo)
    monkeypatch.setattr(find_cameras, "ZedOneCamera", _SpyMono)
    return _SpyStereo, _SpyMono


@pytest.mark.parametrize(
    ("cam_meta", "stereo_built", "mono_built"),
    [(STEREO, 1, 0), (MONO, 0, 1)],
)
def test_save_one_frame_builds_the_backend_matching_the_type(
    spies, monkeypatch, tmp_path, cam_meta, stereo_built, mono_built
):
    """The `type` reported by discovery must pick the matching backend, not default to stereo."""
    stereo, mono = spies
    saved = []
    monkeypatch.setattr("PIL.Image.fromarray", lambda frame: SimpleNamespace(save=saved.append))

    find_cameras._save_one_frame(cam_meta, tmp_path, warmup_s=0)

    assert (len(stereo.instances), len(mono.instances)) == (stereo_built, mono_built)
    built = (stereo.instances + mono.instances)[0]
    assert built.config.serial_number == cam_meta["serial_number"]
    assert saved == [tmp_path / f"{cam_meta['type']}_{cam_meta['serial_number']}.png"]


def test_save_one_frame_disconnects_even_when_the_read_fails(spies, monkeypatch, tmp_path):
    stereo, _ = spies
    disconnected = []
    monkeypatch.setattr(_SpyStereo, "read", lambda self: (_ for _ in ()).throw(RuntimeError("no frame")))
    monkeypatch.setattr(_SpyStereo, "disconnect", lambda self: disconnected.append(self))

    with pytest.raises(RuntimeError, match="no frame"):
        find_cameras._save_one_frame(STEREO, tmp_path, warmup_s=0)

    assert disconnected == stereo.instances


def test_save_images_survives_one_unreadable_camera(fake_finders, monkeypatch, tmp_path, caplog):
    def flaky(cam_meta, output_dir, warmup_s):
        if cam_meta["type"] == "zed":
            raise ConnectionError("camera busy")

    monkeypatch.setattr(find_cameras, "_save_one_frame", flaky)
    monkeypatch.setattr(
        "sys.argv", ["lerobot-find-zed-cameras", "--save-images", "--output-dir", str(tmp_path)]
    )

    with caplog.at_level(logging.ERROR):
        find_cameras.main()

    assert "Could not capture from zed 1" in caplog.text


def test_no_runtime_string_in_the_package_is_non_ascii():
    """A C/POSIX-locale host raises UnicodeEncodeError on non-ASCII output as soon as it is piped.

    Walks the AST rather than matching lines: a `logger.info(` whose f-string sits on the next
    line is invisible to a regex, and that is exactly where the first offender hid.
    """
    import ast
    from pathlib import Path

    package = Path(find_cameras.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any(ord(c) > 127 for c in node.value)
            ):
                offenders.append(f"{path.name}:{node.lineno}: {node.value.strip()[:70]!r}")

    assert not offenders, "non-ASCII in a runtime string:\n" + "\n".join(offenders)
