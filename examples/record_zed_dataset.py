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
"""Record N frames from a ZED straight into a LeRobotDataset — no robot.

Exercises the recording path (ZedCamera -> LeRobotDataset video encode/store) end-to-end
without needing a robot. With ``--depth``, metric depth is stored alongside RGB as a
second stream (uint16 millimetres, encoded as 12-bit video).

Example:
    python examples/zed/record_zed_dataset.py --frames 150 --repo-id local/zed_record_test
    lerobot-dataset-viz --episode-index 0 --repo-id local/zed_record_test
"""

import argparse
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_camera_zed import ZedCamera, ZedCameraConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", type=int, default=None)
    ap.add_argument("--resolution", default="AUTO")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--depth", action="store_true", help="also record metric depth (uint16 mm)")
    ap.add_argument("--depth-mode", default="NEURAL", help="NEURAL|NEURAL_PLUS|NEURAL_LIGHT|ULTRA")
    ap.add_argument("--repo-id", default="local/zed_record_test")
    ap.add_argument("--task", default="zed recording smoke test")
    args = ap.parse_args()

    cam = ZedCamera(
        ZedCameraConfig(
            serial_number=args.serial,
            resolution=args.resolution,
            # fps is NOT passed to the config: ZedCameraConfig requires fps/width/height all-or-none,
            # and the SDK derives the rate from the resolution mode. --fps only paces the record loop.
            use_depth=args.depth,
            depth_mode=args.depth_mode,
        )
    )
    cam.connect()
    height, width = cam.height, cam.width
    fps = int(cam.fps or args.fps)
    print(f"connected: {width}x{height} @ {fps}fps  depth={args.depth}")

    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        },
    }
    if args.depth:
        # `info: {"is_depth_map": True}` is what routes the stream through LeRobot's depth
        # encoder (12-bit, metric-preserving). Without it a 1-channel stream is encoded as
        # ordinary 8-bit video and the millimetre values are destroyed.
        features["observation.images.front_depth"] = {
            "dtype": "video",
            "shape": [height, width, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        robot_type="zed_test",
        use_videos=True,
        image_writer_threads=4,
    )

    print(f"recording {args.frames} frames…")
    for _ in range(args.frames):
        frame = {
            "observation.images.front": cam.async_read(),
            "task": args.task,
        }
        if args.depth:
            frame["observation.images.front_depth"] = cam.read_latest_depth(max_age_ms=500)
        ds.add_frame(frame)
        time.sleep(1.0 / fps)

    ds.save_episode()
    ds.finalize()
    cam.disconnect()
    print(f"done → {ds.root}")
    print(f"visualize:  lerobot-dataset-viz --episode-index 0 --repo-id {args.repo_id}")


if __name__ == "__main__":
    main()
