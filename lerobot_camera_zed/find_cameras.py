#!/usr/bin/env python

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

"""List the Stereolabs cameras the ZED SDK can see, and optionally save one frame from each.

`lerobot-find-cameras` only knows about the backends shipped in LeRobot itself, so a plugin
cannot add a subcommand to it. This is the ZED equivalent:

```shell
lerobot-find-zed-cameras
lerobot-find-zed-cameras --save-images --output-dir outputs/zed-cameras
```

The `serial_number` it prints is what goes into a camera config:
`--robot.cameras='{"front": {"type": "zed", "serial_number": 12345678}}'`
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from lerobot.cameras.configs import ColorMode
from lerobot.utils.utils import init_logging

from .camera_zed import ZedCamera
from .camera_zed_one import ZedOneCamera
from .configuration_zed import ZedCameraConfig
from .configuration_zed_one import ZedOneCameraConfig

logger = logging.getLogger(__name__)


def find_zed_cameras() -> list[dict[str, Any]]:
    """Detect Stereolabs cameras: stereo models under `zed`, the monocular X One under `zed_one`.

    Discovery of one family failing must not hide the other, so each is reported separately.

    Returns:
        One metadata dict per detected camera, as returned by the backends' `find_cameras()`.
    """
    cameras: list[dict[str, Any]] = []
    for finder, label in ((ZedCamera.find_cameras, "stereo"), (ZedOneCamera.find_cameras, "monocular")):
        try:
            cameras.extend(finder())
        except Exception as e:  # noqa: BLE001 — one family failing must not hide the other
            logger.warning(f"ZED {label} discovery failed: {e}")
    return cameras


def _save_one_frame(cam_meta: dict[str, Any], output_dir: Path, warmup_s: int) -> None:
    """Connect to one detected camera, save a single RGB frame, and disconnect."""
    from PIL import Image

    serial = cam_meta["serial_number"]
    if cam_meta["type"] == "zed_one":
        camera = ZedOneCamera(
            ZedOneCameraConfig(serial_number=serial, color_mode=ColorMode.RGB, warmup_s=warmup_s)
        )
    else:
        camera = ZedCamera(ZedCameraConfig(serial_number=serial, color_mode=ColorMode.RGB, warmup_s=warmup_s))

    camera.connect()
    try:
        frame = camera.read()
        path = output_dir / f"{cam_meta['type']}_{serial}.png"
        Image.fromarray(frame).save(path)
        logger.info(f"Saved {path}")
    finally:
        camera.disconnect()


def main() -> None:
    """Entry point for `lerobot-find-zed-cameras`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-images", action="store_true", help="Save one frame per camera.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zed-cameras"),
        help="Where --save-images writes its frames.",
    )
    parser.add_argument("--warmup-s", type=int, default=1, help="Warmup seconds before reading.")
    args = parser.parse_args()

    init_logging()
    cameras = find_zed_cameras()
    if not cameras:
        logger.warning(
            "No ZED cameras were detected. Check the ZED SDK installation (`ZED_Explorer`), the "
            "connection, and that no other process holds the camera."
        )
        return

    print("\n--- Detected ZED cameras ---")
    for cam in cameras:
        print(f"{cam['name']}  type={cam['type']}  serial_number={cam['serial_number']}")
    print("----------------------------\n")

    if not args.save_images:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        try:
            _save_one_frame(cam, args.output_dir, args.warmup_s)
        except Exception as e:  # noqa: BLE001 — one unreadable camera must not stop the others
            logger.error(f"Could not capture from {cam['type']} {cam['serial_number']}: {e}")


if __name__ == "__main__":
    main()
