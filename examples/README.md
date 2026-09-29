# ZED camera examples

Two standalone scripts for the ZED backend (see the [README](../README.md)), useful when bringing
up a new camera before wiring it into a robot.

| Script                  | Purpose                                                                                                                                                                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `smoke_test_zed.py`     | Bench check with no robot and no dataset: discovers the camera, prints resolution/fps/intrinsics, exercises `read()` / `async_read()` / `read_depth()`, and writes `zed_rgb.png` (+ `zed_depth.png` with `--depth`). Run this first when a camera or host is new. |
| `record_zed_dataset.py` | Records N frames from one ZED straight into a `LeRobotDataset` — the recording path without a robot. Useful to verify video encoding and dataset layout on a new host.                                                                                            |

```bash
python examples/smoke_test_zed.py --depth
python examples/record_zed_dataset.py --frames 150 --repo-id local/zed_record_test
lerobot-dataset-viz --episode-index 0 --repo-id local/zed_record_test
```

For real robot data collection use `lerobot-record` with a `{type: zed, ...}` camera config instead —
see the [README](../README.md). For lossless SVO capture and offline metric-depth reclaim, see
`lerobot-zed-export`.
