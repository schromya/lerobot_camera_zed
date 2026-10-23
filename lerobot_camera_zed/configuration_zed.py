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

"""Configuration for the stereo ZED backend (`type: "zed"`)."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation

# NOTE: this file imports NOTHING from pyzed, so it is safe to import at startup
# (in cameras/__init__.py) to register the "zed" draccus choice without pulling
# the ZED SDK dependency.


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for Stereolabs ZED cameras (via the ZED SDK Python API `pyzed`).

    The camera is identified by serial number (recommended); omit it to open the
    first available ZED. `resolution` selects the sensor mode, which fixes the field of view and
    the resolution depth is computed at. `width`/`height` optionally downscale the frames handed
    out; leave them (and `fps`) as None for default configuration.

    Attributes:
        serial_number: ZED serial number. None → first available device.
        resolution: sl.RESOLUTION name; default "AUTO" (picks the camera's native mode).
            USB ZED/ZED2: "HD2K"|"HD1080"|"HD720"|"VGA". GMSL ZED X / X Mini: "HD1200"|"HD1080"|"SVGA"
            (HD720 is INVALID). The ZED X HDR models open only in "AUTO" (native 1920x1536); every
            explicit mode is INVALID. When in doubt, use "AUTO".
        rectified: return the rectified color image (default) or the raw, distorted sensor image.
            Rectification removes lens distortion and is what `intrinsics` describes, so unrectified
            output cannot be used with those intrinsics and is incompatible with `use_depth`.
        depth_mode: sl.DEPTH_MODE name — "NEURAL" | "NEURAL_PLUS" | "NEURAL_LIGHT" | "NONE". The SDK
            still accepts "PERFORMANCE", "QUALITY" and "ULTRA", but deprecates them.
        color_mode: RGB (default) or BGR.
        use_rgb: enable the left color stream. Default True.
        use_depth: enable the metric depth stream (uint16 millimetres). Default False.
        rotation: 0 / 90 / 180 / -90 (`Cv2Rotation`; 270 is not accepted).
        warmup_s: seconds of warmup grabbing before connect() returns.
        svo_record_path: if set, record the camera's full sensor stream to this .svo2 file
            while connected (both eyes, IMU, depth recomputable offline). The SVO gains one
            frame per grab of the background read thread, i.e. the camera's full rate — align
            dataset frames to it via the hardware timestamp (`latest_hw_timestamp_ns`).
        svo_compression: sl.SVO_COMPRESSION_MODE name for `svo_record_path`
            ("H264" | "H265" | "H264_LOSSLESS" | "H265_LOSSLESS" | "LOSSLESS"). The default "H264"
            is lossy; use "LOSSLESS" for a bit-exact archive. On a Jetson Orin Nano the lossy modes
            write empty files for GMSL cameras (no hardware encoder), so use "LOSSLESS" there.
        width / height: output frame size *after* `rotation`; set both or neither. Must not exceed
            the `resolution` mode and must keep its aspect ratio. Depth and `intrinsics` are scaled
            to the same size.
        fps: must be a rate the `resolution` mode supports; None accepts the camera's rate.
    """

    serial_number: int | None = None
    resolution: str = "AUTO"  # AUTO picks each model's native mode (ZED X=HD1080, ZED X HDR=1920x1536, ...)
    rectified: bool = True
    depth_mode: str = "NEURAL"
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    svo_record_path: str | None = None
    svo_compression: str = "H264"

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        if not self.rectified and self.use_depth:
            raise ValueError(
                "`rectified=False` cannot be combined with `use_depth=True`: the ZED SDK computes depth "
                "from the rectified stereo pair, so the depth map would not align with the raw image."
            )

        if (self.width is None) != (self.height is None):
            raise ValueError("`width` and `height` must be set together (or both left unset).")
