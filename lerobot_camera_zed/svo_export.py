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
"""ZED → LeRobot dual exporter.

Two commands (``lerobot-zed-export record`` / ``lerobot-zed-export depth``):

* ``record`` — live capture that writes **both** a LeRobot v3 dataset (left RGB, the stream a VLA
  trains on) **and** a companion **SVO** that losslessly retains the full ZED sensor stream (both eyes,
  depth-on-replay, IMU, intrinsics, hardware timestamps). The two are linked by the ZED hardware
  timestamp, recorded in a ``meta/zed_sources.json`` manifest.
* ``depth`` — offline: replay an SVO and emit **uint16 metric-depth PNG files** (millimetres), aligned to the
  dataset frames.

Why: a LeRobot dataset keeps one lossy video stream per camera key (plus a depth stream when the robot
supports depth cameras). The SVO additionally retains what the dataset cannot: the **stereo pair**, the
IMU, and the raw sensor stream at full rate — so depth can be recomputed offline at a *higher* quality
than was affordable live (record with ``NEURAL_LIGHT``, reclaim with ``NEURAL``), or added after the fact
for a camera that was recorded RGB-only.

For robot data collection, prefer stock ``lerobot-record`` — set ``svo_record_path`` on the camera
config and the SVO sidecar is written transparently. The helpers here cover the camera-only cases:
bench captures with no robot (``record_dual``, ``record_episode``) and offline depth reclaim
(``export_depth_from_svo``). ``observation.state``/``action`` are not available from a camera, so both
recorders accept ``state_fn``/``action_fn`` callables for rigs driven from a script.
"""

import argparse
import contextlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore

_INSTALL_HINT = (
    "The ZED Python API (`pyzed`) is not installed. It ships with the ZED SDK (not PyPI). "
    "Install the ZED SDK 5.4, then run:  python /usr/local/zed/get_python_api.py"
)

try:  # pyzed ships with the ZED SDK, not pip
    import pyzed.sl as sl  # type: ignore
except Exception:  # noqa: BLE001 — defer the failure to first use so `--help`/imports still work
    sl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _require_pyzed() -> None:
    if sl is None:
        raise ImportError(_INSTALL_HINT)


def _open_live(serial: int | None, resolution: str, fps: int | None, depth_mode: str) -> "sl.Camera":
    _require_pyzed()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, resolution)
    if fps:
        init.camera_fps = int(fps)
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    if serial is not None:
        init.set_from_serial_number(int(serial))
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise ConnectionError(f"Failed to open ZED (live): {status}")
    return cam


def _left_rgb(mat: "sl.Mat") -> np.ndarray:
    """ZED LEFT view is BGRA; return a contiguous (H, W, 3) RGB uint8 copy off the reused Mat."""
    bgra = mat.get_data()
    return cv2.cvtColor(np.ascontiguousarray(bgra[:, :, :3]), cv2.COLOR_BGR2RGB)


def record_dual(
    repo_id: str,
    *,
    serial: int | None = None,
    resolution: str = "AUTO",
    fps: int | None = None,
    depth_mode: str = "NEURAL",
    frames: int = 120,
    task: str = "zed dual-export recording",
    image_key: str = "front",
    compression: str = "H264",
    root: str | None = None,
    state_fn: Callable[[], Any] | None = None,
    action_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Record `frames` frames to a LeRobot dataset + a companion SVO, linked by a manifest.

    `state_fn`/`action_fn`, if given, are sampled once per frame and stored as `observation.state` /
    `action` (float32 vectors). Leave them None for a camera-only bench recording.
    """
    cam = _open_live(serial, resolution, fps, depth_mode)
    conf = cam.get_camera_information().camera_configuration
    width, height = int(conf.resolution.width), int(conf.resolution.height)
    cam_fps = int(fps) if fps else int(conf.fps)
    serial = int(cam.get_camera_information().serial_number)

    img_feat = f"observation.images.{image_key}"
    features: dict[str, Any] = {
        img_feat: {"dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channels"]},
    }
    state0 = np.asarray(state_fn(), dtype=np.float32).ravel() if state_fn else None
    action0 = np.asarray(action_fn(), dtype=np.float32).ravel() if action_fn else None
    if state0 is not None:
        # shape must be a tuple: validate_frame compares value.shape (tuple) with `!=`, so a
        # list here always fails (upstream hw_to_dataset_features also emits tuples)
        features["observation.state"] = {"dtype": "float32", "shape": (int(state0.size),), "names": None}
    if action0 is not None:
        features["action"] = {"dtype": "float32", "shape": (int(action0.size),), "names": None}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # needs lerobot[dataset]

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=cam_fps,
        features=features,
        root=root,
        robot_type="zed",
        use_videos=True,
        image_writer_threads=4,  # keep add_frame off the grab loop's critical path
    )

    # Place the SVO inside the dataset dir so the two travel together.
    svo_path = Path(ds.root) / "svo" / "episode_000000.svo2"
    svo_path.parent.mkdir(parents=True, exist_ok=True)
    rec = sl.RecordingParameters(str(svo_path), getattr(sl.SVO_COMPRESSION_MODE, compression))
    status = cam.enable_recording(rec)
    if status != sl.ERROR_CODE.SUCCESS:
        cam.close()
        raise RuntimeError(f"enable_recording failed: {status}")

    logger.info(f"recording {frames} frames @ {width}x{height}@{cam_fps} -> {ds.root} (+ SVO {compression})")
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    frame_meta: list[dict[str, int]] = []
    grabbed = 0
    while grabbed < frames:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:  # SVO frame is auto-written on each success
            continue
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        hw_ts = int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        frame: dict[str, Any] = {img_feat: _left_rgb(mat), "task": task}
        if state_fn:
            frame["observation.state"] = np.asarray(state_fn(), dtype=np.float32).ravel()
        if action_fn:
            frame["action"] = np.asarray(action_fn(), dtype=np.float32).ravel()
        ds.add_frame(frame)
        frame_meta.append({"frame_index": grabbed, "episode_index": 0, "zed_hw_timestamp_ns": hw_ts})
        grabbed += 1

    ds.save_episode()
    ds.finalize()
    cam.disable_recording()
    cam.close()

    # Honest cadence check: the dataset metadata claims `cam_fps`, verify the grabs kept up.
    effective_fps = None
    if len(frame_meta) > 1:
        span_s = (frame_meta[-1]["zed_hw_timestamp_ns"] - frame_meta[0]["zed_hw_timestamp_ns"]) / 1e9
        effective_fps = round((len(frame_meta) - 1) / span_s, 2) if span_s > 0 else None
    if effective_fps is not None and effective_fps < 0.9 * cam_fps:
        logger.warning(
            f"effective capture rate {effective_fps} fps is below the nominal {cam_fps} fps recorded in "
            "dataset metadata - frames were dropped (loop too slow); timestamps in the manifest are authoritative."
        )

    manifest = {
        "effective_fps": effective_fps,
        "svo_files": {"episode_000000": str(svo_path)},
        "serial_number": serial,
        "resolution": resolution,
        "fps": cam_fps,
        "depth_mode": depth_mode,
        "svo_compression": compression,
        "image_key": img_feat,
        "depth_policy": (
            "Metric depth is NOT stored in the LeRobot dataset (v0.5.1 cannot store uint16 depth). "
            "Reclaim it by replaying the SVO with `lerobot-zed-export depth` (MEASURE.DEPTH_U16_MM)."
        ),
        "note_timestamps": (
            "The dataset `timestamp` column is synthetic (frame_index/fps). `zed_hw_timestamp_ns` below "
            "is the true ZED hardware image clock - use it for cross-stream / multi-camera sync."
        ),
        "frames": frame_meta,
    }
    manifest_path = Path(ds.root) / "meta" / "zed_sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote manifest -> {manifest_path}")
    return {"root": str(ds.root), "svo": str(svo_path), "manifest": str(manifest_path), "frames": grabbed}


def record_episode(
    repo_id: str,
    cameras: dict[str, Any],
    *,
    fps: int = 30,
    frames: int = 300,
    task: str = "zed multi-camera recording",
    root: str | None = None,
    state_fn: Callable[[], Any] | None = None,
    action_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Record a LeRobot episode from N cameras (lerobot camera-config dict, like lerobot-record).

    `cameras` maps image keys to `CameraConfig` instances or plain dicts with a `type` field
    (decoded via draccus, exactly like `lerobot-record --robot.cameras`). Each camera becomes
    `observation.images.<key>`. ZED cameras additionally record a lossless SVO sidecar
    (`<root>/svo/<key>_episode_000000.svo2`) at the camera's full rate; dataset frames align to
    SVO frames via the per-frame hardware timestamps stored in `meta/zed_sources.json`
    (`lerobot-zed-export depth --manifest` reclaims metric depth aligned by that clock).
    """
    import draccus  # type: ignore
    from lerobot.cameras.configs import CameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    configs: dict[str, CameraConfig] = {}
    for key, cfg in cameras.items():
        configs[key] = cfg if isinstance(cfg, CameraConfig) else draccus.decode(CameraConfig, cfg)

    cams = make_cameras_from_configs(configs)
    for key, cam in cams.items():
        cam.connect()
        logger.info(f"[{key}] connected: {cam.width}x{cam.height}@{cam.fps}")

    features: dict[str, Any] = {
        f"observation.images.{key}": {
            "dtype": "video",
            "shape": [cam.height, cam.width, 3],
            "names": ["height", "width", "channels"],
        }
        for key, cam in cams.items()
    }
    state0 = np.asarray(state_fn(), dtype=np.float32).ravel() if state_fn else None
    action0 = np.asarray(action_fn(), dtype=np.float32).ravel() if action_fn else None
    if state0 is not None:
        features["observation.state"] = {"dtype": "float32", "shape": (int(state0.size),), "names": None}
    if action0 is not None:
        features["action"] = {"dtype": "float32", "shape": (int(action0.size),), "names": None}

    svo_files: dict[str, str] = {}
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # needs lerobot[dataset]

        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type="zed",
            use_videos=True,
            image_writer_threads=max(4, 2 * len(cams)),
        )

        # Dataset root exists now — start the lossless SVO sidecars (ZED cameras only).
        for key, cam in cams.items():
            explicit = getattr(configs[key], "svo_record_path", None)
            if hasattr(cam, "start_svo_recording") and not getattr(cam, "_svo_recording", False):
                path = explicit or str(Path(ds.root) / "svo" / f"{key}_episode_000000.svo2")
                cam.start_svo_recording(path)
                svo_files[key] = path
            elif explicit:
                svo_files[key] = explicit

        logger.info(f"recording {frames} frames from {list(cams)} -> {ds.root}")
        period = 1.0 / fps
        frame_meta: list[dict[str, Any]] = []
        t_next = time.perf_counter()
        for i in range(frames):
            frame: dict[str, Any] = {"task": task}
            ts_by_key: dict[str, int] = {}
            for key, cam in cams.items():
                frame[f"observation.images.{key}"] = cam.async_read(timeout_ms=1000)
                ts_by_key[key] = int(getattr(cam, "latest_hw_timestamp_ns", 0) or 0)
            if state_fn:
                frame["observation.state"] = np.asarray(state_fn(), dtype=np.float32).ravel()
            if action_fn:
                frame["action"] = np.asarray(action_fn(), dtype=np.float32).ravel()
            ds.add_frame(frame)
            frame_meta.append({"frame_index": i, "episode_index": 0, "zed_hw_timestamp_ns": ts_by_key})
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        ds.save_episode()
        ds.finalize()
    finally:
        for cam in cams.values():
            with contextlib.suppress(Exception):
                cam.disconnect()  # stops SVO recording too

    effective_fps: dict[str, float | None] = {}
    for key in cams:
        ts = [f["zed_hw_timestamp_ns"][key] for f in frame_meta if f["zed_hw_timestamp_ns"].get(key)]
        span = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0
        rate = round((len(ts) - 1) / span, 2) if span > 0 else None
        effective_fps[key] = rate
        if rate is not None and rate < 0.9 * fps:
            logger.warning(
                f"[{key}] effective capture rate {rate} fps < nominal {fps} fps - "
                "manifest timestamps are authoritative."
            )

    manifest = {
        "version": 2,
        "effective_fps": effective_fps,
        "svo_files": svo_files,
        "cameras": {
            key: {
                "type": configs[key].type if hasattr(configs[key], "type") else type(cams[key]).__name__,
                "serial_number": getattr(configs[key], "serial_number", None),
                "width": cams[key].width,
                "height": cams[key].height,
                "rotation": int(getattr(configs[key], "rotation", 0)),
                "fps": cams[key].fps,
                "svo_compression": getattr(configs[key], "svo_compression", None),
            }
            for key in cams
        },
        "depth_policy": (
            "Metric depth is NOT stored in the LeRobot dataset (v0.5.1 cannot store uint16 depth). "
            "Reclaim it per camera with `lerobot-zed-export depth --svo <svo> --manifest "
            "<meta/zed_sources.json> --image-key <key>` (aligned by hardware timestamp)."
        ),
        "note_timestamps": (
            "The dataset `timestamp` column is synthetic (frame_index/fps). `zed_hw_timestamp_ns` maps "
            "each image key to the true ZED hardware clock of the frame stored at that dataset row - "
            "use it for SVO alignment and multi-camera sync."
        ),
        "frames": frame_meta,
    }
    manifest_path = Path(ds.root) / "meta" / "zed_sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"wrote manifest -> {manifest_path}")
    return {
        "root": str(ds.root),
        "svo": svo_files,
        "manifest": str(manifest_path),
        "frames": frames,
        "effective_fps": effective_fps,
    }


def _load_manifest_timestamps(manifest_path: str, image_key: str) -> list[tuple[int, int]]:
    """Return [(frame_index, hw_ts_ns), ...] for one image key from a zed_sources.json (v1 or v2)."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    pairs = []
    for f in m["frames"]:
        ts = f["zed_hw_timestamp_ns"]
        if isinstance(ts, dict):  # v2: per image key
            ts = ts.get(image_key)
        if ts:
            pairs.append((int(f["frame_index"]), int(ts)))
    if not pairs:
        raise ValueError(f"manifest {manifest_path} has no timestamps for image key {image_key!r}")
    return pairs


def _load_manifest_frame_geometry(manifest_path: str, image_key: str) -> tuple[Any, int | None]:
    """Return (sl.Resolution, cv2 rotation code) that make replayed depth match one key's dataset frames."""
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.utils import get_cv2_rotation

    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cam_meta = m.get("cameras", {}).get(image_key, {})
    if "rotation" not in cam_meta or not cam_meta.get("width") or not cam_meta.get("height"):
        return None, None
    rotation = get_cv2_rotation(Cv2Rotation(cam_meta["rotation"]))
    w, h = int(cam_meta["width"]), int(cam_meta["height"])
    if rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        w, h = h, w  # the SDK hands out the frame before rotation
    return sl.Resolution(w, h), rotation


def export_depth_from_svo(
    svo_path: str,
    out_dir: str,
    *,
    depth_mode: str = "NEURAL",
    max_frames: int | None = None,
    manifest: str | None = None,
    image_key: str = "front",
) -> dict[str, Any]:
    """Replay an SVO and write uint16 metric-depth PNG files (mm).

    Without ``manifest``: PNG files are named ``depth_<svo_position>.png`` — valid when the dataset was
    recorded by ``record_dual`` (same grab order, frame_index == svo_position).

    With ``manifest`` (``meta/zed_sources.json``): each SVO frame is matched to the dataset frame with
    the nearest hardware timestamp for ``image_key`` and written as ``depth_<frame_index>.png`` —
    required for ``record_episode`` datasets, where cameras grab at full rate independently of the
    dataset loop. Depth is also resized and rotated to match that key's dataset frames.
    """
    _require_pyzed()
    ts_targets = None
    out_size, rotation = None, None
    if manifest is not None:
        out_size, rotation = _load_manifest_frame_geometry(manifest, image_key)
        pairs = _load_manifest_timestamps(manifest, image_key)
        target_idx = np.array([p[0] for p in pairs])
        target_ts = np.array([p[1] for p in pairs], dtype=np.int64)
        dts = np.diff(np.sort(target_ts))
        tol_ns = int(max(float(np.median(dts)) / 2, 1e6)) if dts.size else int(20e6)
        best: dict[int, tuple[int, str]] = {}  # frame_index -> (abs_err_ns, tmp png path)
        ts_targets = (target_idx, target_ts, tol_ns, best)
    if depth_mode.upper() == "NONE":
        raise ValueError(
            "depth export requires a real depth mode (NEURAL/NEURAL_PLUS/NEURAL_LIGHT/ULTRA), got NONE"
        )
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)  # depth is recomputed on replay
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise ConnectionError(f"Failed to open SVO {svo_path}: {status}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime = sl.RuntimeParameters()
    depth = sl.Mat()
    total = cam.get_svo_number_of_frames()
    written = 0
    dmin, dmax = np.inf, 0
    while True:
        err = cam.grab(runtime)
        if err <= sl.ERROR_CODE.SUCCESS:
            pos = cam.get_svo_position()
            if ts_targets is not None:
                target_idx, target_ts, tol_ns, best = ts_targets
                svo_ts = int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                j = int(np.abs(target_ts - svo_ts).argmin())
                err_ns = abs(int(target_ts[j]) - svo_ts)
                fi = int(target_idx[j])
                if err_ns > tol_ns or (fi in best and err_ns >= best[fi][0]):
                    continue  # no dataset frame claims this SVO frame (or a closer one already did)
                name = f"depth_{fi:06d}.png"
                best[fi] = (err_ns, name)
            else:
                name = f"depth_{pos:06d}.png"
            # uint16 mm, invalid already 0
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH_U16_MM, resolution=out_size)
            d = depth.get_data()
            if d.size == 0:  # retrieve_measure yields an empty Mat if depth is unavailable
                logger.warning(f"empty depth measure at svo position {pos}; skipping frame")
                continue
            if rotation is not None:
                d = cv2.rotate(d, rotation)
            cv2.imwrite(str(out / name), d)
            nz = d[d > 0]
            if nz.size:
                dmin, dmax = min(dmin, int(nz.min())), max(dmax, int(d.max()))
            written += 1
            if max_frames is not None and written >= max_frames:
                break
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        else:
            logger.warning(f"grab returned {err}; stopping SVO replay")
            break
    cam.close()

    (out / "README.txt").write_text(
        "uint16 metric-depth PNG files (millimetres) replayed from the companion SVO.\n"
        "This is a DECLARED, OPT-IN modality - LeRobot v0.5.1 has no metric-depth feature, so depth is\n"
        "kept here (aligned to the dataset by frame index) for the depth-conditioned policy experiment.\n"
        "Note: the final SVO frame is often not replayable (SDK boundary behavior), so the last dataset\n"
        "frame may have no depth sidecar (e.g. 149/150 written).\n",
        encoding="utf-8",
    )
    if ts_targets is not None:
        matched = len(ts_targets[3])
        expected = len(ts_targets[0])
        if matched < expected:
            logger.warning(
                f"{expected - matched}/{expected} dataset frames found no SVO frame within tolerance"
            )
        written = matched
        total = expected
    mm_min = int(dmin) if written and np.isfinite(dmin) else 0  # dmin stays inf if no frame had valid depth
    logger.info(f"wrote {written}/{total} depth PNG files -> {out}  (mm range ~{mm_min}-{dmax})")
    return {"out": str(out), "written": written, "svo_frames": total, "mm_min": mm_min, "mm_max": int(dmax)}


def main() -> None:
    """Entry point for `lerobot-zed-export`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="ZED -> LeRobot dual exporter (SVO + LeRobot dataset).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="live: record RGB -> LeRobot dataset + companion SVO + manifest")
    rec.add_argument("--repo-id", required=True)
    rec.add_argument("--serial", type=int, default=None)
    rec.add_argument("--resolution", default="AUTO")
    rec.add_argument("--fps", type=int, default=None)
    rec.add_argument("--depth-mode", default="NEURAL", help="NEURAL|NEURAL_PLUS|NEURAL_LIGHT|ULTRA")
    rec.add_argument("--frames", type=int, default=120)
    rec.add_argument("--task", default="zed dual-export recording")
    rec.add_argument("--image-key", default="front")
    rec.add_argument("--compression", default="H264", help="H264|H265|H264_LOSSLESS|H265_LOSSLESS|LOSSLESS")
    rec.add_argument("--root", default=None)

    dep = sub.add_parser("depth", help="offline: replay an SVO -> uint16 metric-depth PNG sidecar")
    dep.add_argument("--svo", required=True)
    dep.add_argument("--out", required=True)
    dep.add_argument("--depth-mode", default="NEURAL")
    dep.add_argument("--max-frames", type=int, default=None)
    dep.add_argument(
        "--manifest",
        default=None,
        help="meta/zed_sources.json - align PNG names to dataset frame_index by hardware "
        "timestamp (required for record_episode/multi-camera datasets)",
    )
    dep.add_argument("--image-key", default="front", help="image key in the manifest (with --manifest)")

    args = parser.parse_args()
    if args.cmd == "record":
        out = record_dual(
            args.repo_id,
            serial=args.serial,
            resolution=args.resolution,
            fps=args.fps,
            depth_mode=args.depth_mode,
            frames=args.frames,
            task=args.task,
            image_key=args.image_key,
            compression=args.compression,
            root=args.root,
        )
        print(f"done -> dataset {out['root']}\n  SVO      {out['svo']}\n  manifest {out['manifest']}")
        print(f"visualize:  lerobot-dataset-viz --episode-index 0 --repo-id {args.repo_id}")
        print(
            f"reclaim depth:  lerobot-zed-export depth --svo {out['svo']} --out {out['root']}/depth/{args.image_key}"
        )
    elif args.cmd == "depth":
        out = export_depth_from_svo(
            args.svo,
            args.out,
            depth_mode=args.depth_mode,
            max_frames=args.max_frames,
            manifest=args.manifest,
            image_key=args.image_key,
        )
        print(
            f"wrote {out['written']}/{out['svo_frames']} depth PNG files -> {out['out']}  (mm ~{out['mm_min']}-{out['mm_max']})"
        )


if __name__ == "__main__":
    main()
