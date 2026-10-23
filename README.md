# lerobot_camera_zed

Stereolabs **ZED camera plugin for [LeRobot](https://github.com/huggingface/lerobot)**.

Adds two camera backends that plug into `lerobot-record`, `lerobot-teleoperate`, `lerobot-calibrate`,
`lerobot-replay`, `lerobot-rollout`, `lerobot-eval` and the async-inference client with no changes to
LeRobot itself:

| Camera type | Class           | Hardware                                                              |
| ----------- | --------------- | --------------------------------------------------------------------- |
| `zed`       | `ZedCamera`     | Stereo: ZED 2 / 2i / Mini (USB), ZED X / X Mini / X HDR family (GMSL) |
| `zed_one`   | `ZedOneCamera`  | Monocular ZED X One (GMSL)                                            |

On top of the standard LeRobot camera API it gives you metric depth (`uint16` millimetres, the same
shape/dtype contract as the RealSense backend, so ZED depth datasets are interchangeable with
RealSense ones downstream), the camera's hardware image timestamp (`latest_hw_timestamp_ns`) for
multi-camera and robot-state alignment, and SVO recording/offline depth export.

## Install

```bash
pip install lerobot_camera_zed
```

Then install the **ZED SDK 5.4 or newer** and generate `pyzed` into the same environment as LeRobot.
The Python API ships with the SDK, it is **not** on PyPI — the PyPI project named `pyzed` is an
unrelated package, which is why this plugin does not declare it as a dependency:

```bash
python /usr/local/zed/get_python_api.py        # provides `pyzed.sl`
```

LeRobot discovers the plugin automatically by package name, so nothing else is needed. Verify:

```bash
lerobot-find-zed-cameras
```

On Jetson, the `NEURAL` depth modes also need the ZED AI and TensorRT libraries on the library path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

> [!WARNING]
> `pyzed` 5.4 requires `numpy>=2`. If the OpenCV in your environment is a numpy-1 build, the import
> will clash — install matching wheels in the same virtual environment.

## Quickstart

```bash
lerobot-find-zed-cameras          # prints the serial number to use below

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ front: {type: zed, serial_number: 12345678, use_depth: true, width: 1920, height: 1200, fps: 30} }' \
  --dataset.repo_id=${HF_USER}/my-zed-dataset \
  --dataset.num_episodes=5
```

```python
from lerobot_camera_zed import ZedCamera, ZedCameraConfig

config = ZedCameraConfig(serial_number=12345678, use_depth=True)
with ZedCamera(config) as camera:
    rgb = camera.read_latest()
    depth = camera.read_latest_depth()  # same grab as `rgb`, so the two stay pixel-aligned
```

## Choose a resolution

`resolution` selects the sensor mode, which fixes the field of view and the resolution depth is
computed at. Leave it at `"AUTO"` unless you have a reason not to: `AUTO` opens each model in its
native mode, and the supported mode list differs per model. `width`/`height` optionally downscale
the frames handed out.

| Camera family               | Supported modes                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| USB (ZED, ZED 2i, ZED Mini) | `HD2K`, `HD1080`, `HD720`, `VGA`                                                                       |
| GMSL (ZED X, ZED X Mini)    | `HD1200`, `HD1080`, `SVGA` — **`HD720` is rejected**                                                   |
| ZED X HDR family            | **Only open at their native `1920x1536`** — `HD1080`, `HD1200` and `SVGA` are all rejected, use `AUTO` |

> [!TIP]
> Standalone, leave `fps`, `width` and `height` unset for default configuration. `width` and
> `height` are the output frame size *after* `rotation`; set both or neither. They must not exceed
> the `resolution` mode and must keep its aspect ratio, e.g. `resolution: HD1200, width: 640,
> height: 400`. Depth and `intrinsics` are scaled to the same size. `fps` is independent and is
> *requested* from the SDK, so it is honoured only when the mode supports that rate: `HD2K` on a
> ZED 2i runs at 15 fps and nothing lower, and asking for less makes `connect()` raise rather than
> silently capture at 15.

> [!WARNING]
> Inside a **robot** config you must set `width`, `height` and `fps` explicitly (for example
> `width: 1920, height: 1200, fps: 30`), because robots publish their observation features before
> the cameras connect, so the shape cannot be filled in at connect time. `fps` must be a rate the
> mode supports and `width`/`height` must fit the mode as above, otherwise `connect()` raises.

## Configuration reference

`ZedCameraConfig` accepts the following fields:

| Field                      | Default    | Notes                                                                                                                                                                   |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `serial_number`            | `None`     | `None` opens the first ZED found                                                                                                                                        |
| `resolution`               | `"AUTO"`   | `AUTO` picks the model's native mode. `HD2K` / `HD1080` / `HD1200` / `SVGA` / `HD720` / `VGA` — see the per-model traps above                                           |
| `rectified`                | `True`     | `True` returns the rectified image (lens distortion removed); `False` returns the raw sensor image. `intrinsics` follows the choice. Cannot be combined with `use_depth` |
| `depth_mode`               | `"NEURAL"` | `NEURAL` / `NEURAL_PLUS` / `NEURAL_LIGHT` / `NONE` (the SDK still accepts `PERFORMANCE`, `QUALITY` and `ULTRA` but deprecates them)                                     |
| `use_rgb`                  | `True`     | Left color stream                                                                                                                                                       |
| `use_depth`                | `False`    | Depth as `(H, W, 1)` `uint16` millimetres, the unit LeRobot's depth path expects                                                                                        |
| `color_mode`               | `RGB`      | `RGB` or `BGR`                                                                                                                                                          |
| `rotation`                 | `0`        | `0` / `90` / `180` / `-90` (`Cv2Rotation`; `270` is not accepted)                                                                                                       |
| `warmup_s`                 | `1`        | Seconds of warmup grabbing before `connect()` returns                                                                                                                   |
| `svo_record_path`          | `None`     | Record the full sensor stream to this `.svo2` file while connected — see [SVO recording](#svo-recording)                                                                |
| `svo_compression`          | `"H264"`   | `H264` / `H265` / `H264_LOSSLESS` / `H265_LOSSLESS` / `LOSSLESS`                                                                                                        |
| `fps` / `width` / `height` | `None`     | Inherited fields — `width`/`height`: output size after `rotation`, set both or neither, must fit the mode and keep its aspect ratio; `fps`: a rate the mode supports |

`ZedOneCameraConfig` is the same minus `use_rgb`, `use_depth` and `depth_mode`.

## Rectified or raw images

By default the backend returns the **rectified** left image: the SDK has removed lens distortion, so
straight lines in the world are straight in the image. Set `rectified: false` to receive the **raw**
sensor image instead, which is what you want when you intend to do your own undistortion, calibrate
against the raw optics, or feed a model trained on unrectified data.

```python
config = ZedCameraConfig(serial_number=12345678, rectified=False)
```

`intrinsics` follows the choice: rectified output reports the rectified calibration, raw output
reports the raw one. On a ZED 2i the two differ substantially (measured `fx` 732.6 rectified versus
531.9 raw), so do not mix one with the other.

> [!WARNING]
> `rectified: false` cannot be combined with `use_depth: true` — the SDK computes depth from the
> rectified stereo pair, so a depth map would not align with the raw image. The config raises.

## Depth

With `use_depth: true`, `read_depth()` and `read_latest_depth()` return metric depth as `(H, W, 1)`
`uint16` millimetres, and depth-aware robots record it into the dataset as a sibling feature
(`observation.images.front_depth` next to `observation.images.front`). LeRobot stores it as a 12-bit
video stream that decodes back to millimetres.

Two encoder defaults are worth knowing about, since they come from
[LeRobot's depth encoder](https://huggingface.co/docs/lerobot/main/en/video_encoding_parameters#depth-streams)
rather than from this plugin:

- The quantization range is `depth_min` 0.01 m to `depth_max` 10 m. A ZED sees much farther than
  that and everything beyond saturates, so raise `depth_max` through the dataset's `depth_encoder`
  settings if your scene is deeper. Precision is log-distributed, so it is finest close up.
- Invalid pixels (0 mm) come back as `depth_min`, not 0. The ZED marks unmatched pixels as 0 before
  encoding, so treat values sitting at the floor as "no return" rather than "1 cm away".

> [!WARNING]
> A script that builds its dataset features by hand must tag the depth stream with
> `"info": {"is_depth_map": True}`. Without that flag a single-channel stream is encoded as ordinary
> 8-bit video and the metric values are lost.

The backend exposes the **left** color view only, because LeRobot models one image stream per camera
key rather than a stereo pair.

`read()` and `read_depth()` each wait for the _next_ grab, so calling them back to back returns two
different captures. For a colour/depth pair from the same grab, use the `read_latest*` methods, as in
the quickstart above.

## SVO recording

Set `svo_record_path` to capture the camera's full sensor stream (both eyes, IMU, depth recomputable
offline) to a `.svo2` file for as long as the camera is connected, or call `start_svo_recording(path)`
at runtime. The SVO gains one frame per grab of the background read thread, i.e. the camera's full
rate, so align dataset frames to it through `latest_hw_timestamp_ns`.

```python
config = ZedCameraConfig(
    serial_number=64058900,
    use_depth=True,
    svo_record_path="episode_000.svo2",
    svo_compression="LOSSLESS",  # this serial is a GMSL ZED X HDR — see the warning below
)
with ZedCamera(config) as camera:
    frame = camera.read()
    depth = camera.read_depth()
```

> [!WARNING]
> The default `H264` (and `H265`, `*_LOSSLESS`) modes need a hardware encoder. On a Jetson **without
> an NVENC node** they write a header-only, frame-less file for GMSL cameras while
> `enable_recording()` still reports success; use `svo_compression: LOSSLESS` there. The backend
> detects this by probing for `/dev/*nvenc*` rather than by assuming every Jetson lacks one — a
> Jetson AGX Orin records H264 over GMSL perfectly well (measured: 22.8 MB versus 570.7 MB for the
> same 5 seconds in LOSSLESS), and is not warned about. USB cameras fall back to CPU encoding and
> record normally.

### Reclaim metric depth offline

`lerobot-zed-export depth` replays a recorded SVO and writes `uint16` millimetre PNG files:

```bash
lerobot-zed-export depth --svo episode_000.svo2 --out depth/front --depth-mode NEURAL
```

This is how you get a higher-quality depth mode than you could afford live (record with
`NEURAL_LIGHT`, reclaim with `NEURAL`), or depth for a camera that was recorded without it.

For multi-camera recordings, pass the manifest so frames are matched by hardware clock instead of
position, and name the camera whose stream you are reclaiming:

```bash
lerobot-zed-export depth \
  --svo front.svo2 \
  --manifest ~/.cache/huggingface/lerobot/local/zed_demo/meta/zed_sources.json \
  --image-key front \
  --out depth/front
```

Each ZED camera writes its own SVO sidecar, and `meta/zed_sources.json` stores per-frame hardware
timestamps per image key. Rows that cannot be matched — after a camera hiccup, for instance — are
reported rather than silently duplicated.

## Monocular ZED X One

The ZED X One is a single-sensor camera, so the SDK drives it through a different API
(`sl.CameraOne`). It gets its own backend class and camera type:

```python
from lerobot_camera_zed import ZedOneCamera, ZedOneCameraConfig

cam = ZedOneCamera(ZedOneCameraConfig(serial_number=12345678))
cam.connect()
rgb = cam.read()  # (H, W, 3) uint8 — 1920x1200 native on the X One GS
cam.disconnect()
```

In a robot config, use `type: zed_one` (and remember the explicit width/height/fps rule):

```bash
--robot.cameras='{ side: {type: zed_one, serial_number: 12345678, width: 1920, height: 1200, fps: 30} }'
```

Everything from the stereo backend carries over — the background capture thread, `read()` /
`async_read()` / `read_latest()`, `rectified`, rotation, colour mode, the hardware timestamp and SVO
recording — with one exception:

> [!WARNING]
> A single sensor cannot triangulate, so there is **no depth**: `ZedOneCameraConfig` has no
> `use_depth` / `depth_mode`, and `read_depth()`, `async_read_depth()` and `read_latest_depth()`
> raise `NotImplementedError`.

`lerobot-find-zed-cameras` lists monocular and stereo cameras together, each under its own `type`
(`zed` or `zed_one`). The SDK keeps the two device lists separate, so a stereo camera never shows up
as a monocular entry, and both kinds can capture concurrently in the same rig.

## Discovering cameras

`lerobot-find-cameras` only knows about the backends shipped inside LeRobot, and a plugin cannot add
a subcommand to it, so this package ships its own equivalent:

```bash
lerobot-find-zed-cameras                       # list every ZED the SDK can see
lerobot-find-zed-cameras --save-images         # plus one PNG per camera, for placement checks
```

## Tested hardware

All rows below were exercised with ZED SDK 5.4.

| Camera               | Host              | Mode                           | Exercised                                                       |
| -------------------- | ----------------- | ------------------------------ | ----------------------------------------------------------------- |
| ZED 2i               | Desktop x86 + RTX | HD720 / HD1080 / VGA @ 30      | Discovery, RGB + depth, `lerobot-record` and Rerun visualization |
| 2x ZED X Mini (GMSL) | Jetson Orin NX    | AUTO `1920x1200` @ 30          | Concurrent dual capture, `lerobot-record` with a bimanual robot  |
| ZED X                | Jetson AGX Orin   | HD1080 @ 30 + NEURAL           | RGB `(1080, 1920, 3)`, depth `(1080, 1920, 1)` `uint16` mm       |
| ZED X HDR            | Jetson Orin Nano  | AUTO `1920x1536` @ 30 + NEURAL | RGB `(1536, 1920, 3)`, depth `(1536, 1920, 1)` `uint16` mm       |
| ZED X One GS (mono)  | Jetson Orin NX    | AUTO `1920x1200` @ 30          | `ZedOneCamera` discovery, RGB, raw and rectified calibration     |

## Grab warnings

The SDK reports negative `ERROR_CODE`s from `grab()` as warnings. The backend logs each one once per
streak (`ZedCamera(...) grab() warning: ...`) and keeps the frame, except for `CORRUPTED_FRAME`
(green/purple image flagged by the SDK's validity check) and `CAMERA_REBOOTING`, which carry no usable
image and are dropped: `read()` waits for the next good frame and `read_latest()` ages out. Repeated
`CORRUPTED_FRAME` warnings point at cabling or driver problems, not at the backend. The SDK's own
console output is silenced unless the `lerobot_camera_zed` logger is at `DEBUG`.

## Embedded hosts (Jetson)

Guidance learned while recording with two cameras on a Jetson Orin NX:

- Keep `--display_data` **off** while recording. Live visualization on the iGPU starves the capture
  pipeline and produces choppy video; stream to another machine with `--display_mode=foxglove`
  instead.
- Two cameras at `1920x1200` push the record loop close to its budget. `SVGA` (960x600) leaves
  comfortable headroom and is ample for wrist views.
- Post-episode video encoding saturates the CPU. If your robot streams telemetry over the network,
  set `--dataset.video_encoding_batch_size` greater than 1 so encoding happens once at the end of the
  session instead of between episodes.
- `NEURAL_PLUS` needs the SDK to build a TensorRT engine on first use. On the 4 GB Orin Nano that
  build does not complete — use `NEURAL` or `NEURAL_LIGHT` there.

> [!WARNING]
> Reading datasets back on a Jetson may fail in `torchcodec` with
> `libtorchcodec_core*.so: could not load`. This is unrelated to this plugin: the aarch64
> `torchcodec` wheel requires `torch>=2.11`, and PyPI's aarch64 torch is built against CUDA 13 while
> JetPack 6 ships CUDA 12.x, so the `libnvrtc.so.13` dependency is missing. Recording is unaffected —
> pass `video_backend="pyav"` when reading, or install NVIDIA's JetPack-specific torch build.

## Examples

See [`examples/`](examples) for a standalone smoke test and a dataset recorder.

## Development

```bash
pip install -e ".[test]"
pytest              # 207 tests, no ZED SDK and no hardware required (they run against a mock)
ruff check . && ruff format --check .
mypy
```

## License

Apache-2.0. See [LICENSE](LICENSE).
