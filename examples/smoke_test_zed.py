#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""M0 smoke test — bench-verify the ZED backend with just a camera (no robot).

Run on a machine with a ZED attached and the ZED SDK 5.4 Python API installed:
    python smoke_test_zed.py
Writes zed_rgb.png (+ zed_depth.png if --depth) so you can eyeball the frames.
"""

import argparse

import numpy as np
from PIL import Image

from lerobot_camera_zed import ZedCamera, ZedCameraConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", type=int, default=None)
    ap.add_argument(
        "--resolution",
        default="AUTO",
        help="AUTO picks the native mode. ZED X=HD1080/HD1200/SVGA; ZED X HDR=AUTO only (1920x1536); HD720 is USB-only.",
    )
    ap.add_argument("--depth", action="store_true")
    args = ap.parse_args()

    print("Detected ZED cameras:", ZedCamera.find_cameras())

    cfg = ZedCameraConfig(
        serial_number=args.serial,
        resolution=args.resolution,
        use_depth=args.depth,
        depth_mode="NEURAL",
    )
    cam = ZedCamera(cfg)
    cam.connect()
    try:
        print(f"connected: {cam.width}x{cam.height} @ {cam.fps}fps  intrinsics={cam.intrinsics}")

        rgb = cam.read()
        print("read() RGB:", rgb.shape, rgb.dtype)
        assert rgb.shape == (cam.height, cam.width, 3) and rgb.dtype == np.uint8
        Image.fromarray(rgb).save("zed_rgb.png")

        a = cam.async_read()
        print("async_read() RGB:", a.shape, "hw_ts_ns:", cam.latest_hw_timestamp_ns)

        if args.depth:
            d = cam.read_depth()
            print("read_depth() uint16 mm:", d.shape, d.dtype, "range:", int(d.min()), int(d.max()))
            assert d.shape == (cam.height, cam.width, 1) and d.dtype == np.uint16
            vis = (np.clip(d[..., 0], 0, 4000) / 4000 * 255).astype(np.uint8)
            Image.fromarray(vis).save("zed_depth.png")

        print("OK — wrote zed_rgb.png" + (" + zed_depth.png" if args.depth else ""))
    finally:
        cam.disconnect()


if __name__ == "__main__":
    main()
