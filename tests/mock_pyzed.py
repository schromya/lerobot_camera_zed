# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Stand-in for the `pyzed.sl` module used by the ZED camera tests.

`pyzed` ships with the ZED SDK and is not installable from PyPI, so it cannot be a CI dependency.
`lerobot_camera_zed` imports it behind a try/except; the tests replace the module handle with the
namespace built by `make_fake_sl`, which mimics the small subset of the SDK the backends touch. The
fake cameras never assert: they record every call so a test can check them on the main thread.
"""

import contextlib
import time
from enum import Enum, IntEnum
from types import SimpleNamespace

import numpy as np

DEFAULT_SERIAL = 12345678
STEREO_SERIAL = 87654321
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 60
# Synthetic left-view pixel, in the BGRA order the ZED SDK returns for sl.VIEW.LEFT.
BGRA_PIXEL = (10, 20, 30, 255)
# MEASURE.DEPTH_U16_MM hands back uint16 millimetres with 0 marking an invalid measurement.
DEPTH_MM = 1500
INVALID_DEPTH = 0
# The rectified and raw calibrations differ, as on real hardware.
RECTIFIED_FX = 700.0
UNRECTIFIED_FX = 530.0


class FakeResolution(Enum):
    AUTO = 0
    HD2K = 1
    HD1200 = 2
    HD1080 = 3
    HD720 = 4
    SVGA = 5
    VGA = 6
    LAST = 7


class FakeDepthMode(Enum):
    # Mirrors sl.DEPTH_MODE in SDK 5.4; PERFORMANCE, QUALITY and ULTRA are deprecated there.
    NONE = 0
    PERFORMANCE = 1
    QUALITY = 2
    ULTRA = 3
    NEURAL_LIGHT = 4
    NEURAL = 5
    NEURAL_PLUS = 6
    CUSTOM = 7
    LAST = 8


class FakeErrorCode(IntEnum):
    # IntEnum, like sl.ERROR_CODE: negative values are warnings, positive values are errors.
    SENSOR_CONFIGURATION_CHANGED = -6
    POTENTIAL_CALIBRATION_ISSUE = -5
    CONFIGURATION_FALLBACK = -4
    CORRUPTED_FRAME = -2
    CAMERA_REBOOTING = -1
    SUCCESS = 0
    FAILURE = 1
    CAMERA_NOT_DETECTED = 2
    INVALID_RESOLUTION = 3


class FakeSvoCompressionMode(Enum):
    LOSSLESS = 0
    H264 = 1
    H265 = 2
    H264_LOSSLESS = 3
    H265_LOSSLESS = 4
    LAST = 5


class FakeInputType(Enum):
    USB = 0
    GMSL = 1
    SVO = 2
    STREAM = 3


class FakeUnit(Enum):
    MILLIMETER = 0
    CENTIMETER = 1
    METER = 2


class FakeView(Enum):
    LEFT = 0
    RIGHT = 1
    LEFT_UNRECTIFIED = 2


class FakeMeasure(Enum):
    DEPTH = 0
    XYZ = 1
    DEPTH_U16_MM = 2


class FakeTimeReference(Enum):
    IMAGE = 0
    CURRENT = 1


class FakeSize:
    """Stand-in for `sl.Resolution` (a width/height pair)."""

    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height


class FakeMat:
    """Stand-in for `sl.Mat`: a reusable buffer the SDK writes each frame into."""

    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class FakeTimestamp:
    def __init__(self, nanoseconds: int):
        self._nanoseconds = nanoseconds

    def get_nanoseconds(self) -> int:
        return self._nanoseconds


class FakeInitParameters:
    """Stand-in for `sl.InitParameters`, the *stereo* open parameters."""

    def __init__(self):
        self.camera_resolution = FakeResolution.AUTO
        self.camera_fps = 0
        self.depth_mode = FakeDepthMode.NEURAL
        # the remaining defaults mirror pyzed 5.4 (they differ from the C++ header for validity_check)
        self.coordinate_units = FakeUnit.MILLIMETER
        self.sdk_verbose = 1
        self.enable_image_validity_check = 0
        self.serial_number = None

    def set_from_serial_number(self, serial_number):
        self.serial_number = int(serial_number)


class FakeInitParametersOne:
    """Stand-in for `sl.InitParametersOne`: like `sl.InitParameters` but with no `depth_mode`."""

    def __init__(self):
        self.camera_resolution = FakeResolution.AUTO
        self.camera_fps = 0
        self.coordinate_units = FakeUnit.MILLIMETER
        self.sdk_verbose = 1
        self.serial_number = None

    def set_from_serial_number(self, serial_number):
        self.serial_number = int(serial_number)


class FakeRecordingParameters:
    """Stand-in for `sl.RecordingParameters`."""

    def __init__(self, video_filename, compression_mode):
        self.video_filename = video_filename
        self.compression_mode = compression_mode


class FakeRuntimeParameters:
    """Exists on the real `sl` module (for `sl.Camera.grab`), but `sl.CameraOne.grab` takes none."""


def fake_device(camera_model="ZED X", serial_number=DEFAULT_SERIAL, path="/dev/i2c-14"):
    """Stand-in for `sl.DeviceProperties`, as returned by the SDK device lists."""
    return SimpleNamespace(camera_model=camera_model, serial_number=serial_number, path=path)


class _FakeCameraBase:
    """Behaviour shared by the stereo and monocular fakes: a BGRA image served off one buffer."""

    def __init__(
        self,
        width,
        height,
        fps,
        open_status,
        grab_status,
        grab_delay_s,
        served_size,
        enable_recording_status=FakeErrorCode.SUCCESS,
        input_type=FakeInputType.USB,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.open_status = open_status
        self.grab_status = grab_status
        self.grab_delay_s = grab_delay_s
        self.enable_recording_status = enable_recording_status
        self.input_type = input_type

        self.recording_parameters = None
        self.is_recording = False
        self.init_parameters = None
        self.opened = False
        self.close_count = 0
        self.grab_count = 0
        # Recorded rather than asserted: an assert inside a fake would fire on the read thread.
        self.requested_views = []
        self.requested_measures = []
        # (width, height) asked of retrieve_image / retrieve_measure, None when left to the SDK
        self.requested_sizes = []
        self.time_references = []

        # The SDK hands back a view on an internal buffer, so allocate these once. `served_size`
        # lets a test hand out frames that do not match the mode the camera reports.
        self.served_size = served_size
        served_w, served_h = served_size or (width, height)
        self._color = self._make_color(served_w, served_h)
        self._depth = self._make_depth(served_w, served_h)
        self._resized = {}  # (kind, w, h) -> buffer, for retrieves below the camera mode

    @staticmethod
    def _make_color(w, h):
        color = np.zeros((h, w, 4), dtype=np.uint8)
        color[:, :] = BGRA_PIXEL
        return color

    @staticmethod
    def _make_depth(w, h):
        depth = np.full((h, w), DEPTH_MM, dtype=np.uint16)
        # The SDK collapses occluded / too-far / too-close pixels to 0 in DEPTH_U16_MM.
        depth[0, 0:3] = INVALID_DEPTH
        return depth

    def _serve(self, kind, resolution):
        """The buffer for a retrieve: the camera mode, or the downscaled size that was asked for."""
        size = None if resolution is None else (resolution.width, resolution.height)
        self.requested_sizes.append(size)
        native = self._color if kind == "color" else self._depth
        if self.served_size is not None or size is None or size == (self.width, self.height):
            return native
        if size[0] > self.width or size[1] > self.height:
            raise AssertionError(f"retrieve at {size} exceeds the camera mode {(self.width, self.height)}")
        key = (kind, *size)
        if key not in self._resized:
            make = self._make_color if kind == "color" else self._make_depth
            self._resized[key] = make(*size)
        return self._resized[key]

    def _calibration(self, resizer):
        """Rectified and raw pinhole parameters, scaled like the SDK does for a `resizer` size."""
        sx = resizer.width / self.width if resizer is not None else 1.0
        sy = resizer.height / self.height if resizer is not None else 1.0
        rectified = SimpleNamespace(
            fx=RECTIFIED_FX * sx, fy=701.0 * sy, cx=self.width / 2 * sx, cy=self.height / 2 * sy
        )
        raw = SimpleNamespace(
            fx=UNRECTIFIED_FX * sx, fy=531.0 * sy, cx=(self.width / 2 + 9) * sx, cy=(self.height / 2 + 7) * sy
        )
        size = (resizer.width, resizer.height) if resizer is not None else (self.width, self.height)
        return SimpleNamespace(width=size[0], height=size[1]), rectified, raw

    def open(self, init_parameters):
        self.init_parameters = init_parameters
        self.opened = self.open_status <= FakeErrorCode.SUCCESS
        return self.open_status

    def is_opened(self):
        return self.opened

    def _grab(self):
        self.grab_count += 1
        time.sleep(self.grab_delay_s)  # keep the background read thread from spinning flat out
        return self.grab_status

    def retrieve_image(self, mat, view, mem_type=None, resolution=None):
        self.requested_views.append(view)
        mat.data = self._serve("color", resolution)
        return FakeErrorCode.SUCCESS

    def get_timestamp(self, time_reference):
        self.time_references.append(time_reference)
        return FakeTimestamp(1_700_000_000_000_000_000 + self.grab_count)

    def enable_recording(self, recording_parameters):
        self.recording_parameters = recording_parameters
        self.is_recording = self.enable_recording_status == FakeErrorCode.SUCCESS
        return self.enable_recording_status

    def disable_recording(self):
        self.is_recording = False

    def close(self):
        self.opened = False
        self.close_count += 1


class FakeCamera(_FakeCameraBase):
    """Stand-in for `sl.Camera` (stereo): BGRA colour plus uint16 millimetre depth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One entry per grab() call, holding the RuntimeParameters it received.
        self.grab_runtime_parameters = []

    def get_camera_information(self, resizer=None):
        resolution, rectified, raw = self._calibration(resizer)
        return SimpleNamespace(
            input_type=self.input_type,
            camera_configuration=SimpleNamespace(
                resolution=resolution,
                fps=self.fps,
                calibration_parameters=SimpleNamespace(left_cam=rectified),
                calibration_parameters_raw=SimpleNamespace(left_cam=raw),
            ),
        )

    def grab(self, runtime_parameters):
        self.grab_runtime_parameters.append(runtime_parameters)
        return self._grab()

    def retrieve_measure(self, mat, measure, mem_type=None, resolution=None):
        self.requested_measures.append(measure)
        mat.data = self._serve("depth", resolution)
        return FakeErrorCode.SUCCESS


class FakeCameraOne(_FakeCameraBase):
    """Stand-in for `sl.CameraOne` (single sensor): BGRA colour only, flat calibration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One entry per grab() call, holding how many positional arguments it received: the
        # monocular API has no RuntimeParameters, so these must all be 0.
        self.grab_arg_counts = []

    def get_camera_information(self, resizer=None):
        # `CameraOne` exposes a *flat* calibration (no `.left_cam`): there is a single sensor.
        resolution, rectified, raw = self._calibration(resizer)
        return SimpleNamespace(
            input_type=self.input_type,
            camera_configuration=SimpleNamespace(
                resolution=resolution,
                fps=self.fps,
                calibration_parameters=rectified,
                calibration_parameters_raw=raw,
            ),
        )

    def grab(self, *args):
        self.grab_arg_counts.append(len(args))
        return self._grab()


def make_fake_sl(
    width=CAPTURE_WIDTH,
    height=CAPTURE_HEIGHT,
    fps=CAPTURE_FPS,
    open_status=FakeErrorCode.SUCCESS,
    grab_status=FakeErrorCode.SUCCESS,
    grab_delay_s=0.002,
    served_size=None,
    stereo_device_list=None,
    mono_device_list=None,
    enable_recording_status=FakeErrorCode.SUCCESS,
    input_type=FakeInputType.USB,
):
    """Build a stand-in for the `pyzed.sl` module.

    The returned namespace also exposes `cameras`, the list of fake `sl.Camera` / `sl.CameraOne`
    objects the backend instantiated, so tests can assert on what was passed to the SDK.
    """
    fake = SimpleNamespace(
        RESOLUTION=FakeResolution,
        DEPTH_MODE=FakeDepthMode,
        SVO_COMPRESSION_MODE=FakeSvoCompressionMode,
        INPUT_TYPE=FakeInputType,
        ERROR_CODE=FakeErrorCode,
        UNIT=FakeUnit,
        VIEW=FakeView,
        MEASURE=FakeMeasure,
        TIME_REFERENCE=FakeTimeReference,
        Mat=FakeMat,
        Resolution=FakeSize,
        InitParameters=FakeInitParameters,
        InitParametersOne=FakeInitParametersOne,
        RuntimeParameters=FakeRuntimeParameters,
        RecordingParameters=FakeRecordingParameters,
        cameras=[],
    )
    stereo_devices = [fake_device()] if stereo_device_list is None else stereo_device_list
    mono_devices = [fake_device("ZED X One GS")] if mono_device_list is None else mono_device_list
    camera_kwargs = {
        "width": width,
        "height": height,
        "fps": fps,
        "open_status": open_status,
        "grab_status": grab_status,
        "grab_delay_s": grab_delay_s,
        "served_size": served_size,
        "enable_recording_status": enable_recording_status,
        "input_type": input_type,
    }

    class Camera(FakeCamera):
        def __init__(self):
            super().__init__(**camera_kwargs)
            fake.cameras.append(self)

        @staticmethod
        def get_device_list():
            return stereo_devices

    class CameraOne(FakeCameraOne):
        def __init__(self):
            super().__init__(**camera_kwargs)
            fake.cameras.append(self)

        @staticmethod
        def get_device_list():
            return mono_devices

    fake.Camera = Camera
    fake.CameraOne = CameraOne
    return fake


class ZedTestHarness:
    """Patches the backend's `pyzed.sl` handle and hands out cameras that always get cleaned up.

    `modules` are the backend modules whose `sl` global must be replaced: the shared machinery
    resolves `sl` in `base_zed`, the model-specific calls in `camera_zed` / `camera_zed_one`.
    """

    def __init__(self, monkeypatch, modules, camera_cls, config_cls):
        self._monkeypatch = monkeypatch
        self._modules = modules
        self._camera_cls = camera_cls
        self._config_cls = config_cls
        self._cameras = []
        self.sl = None
        self.patch_sl()

    def patch_sl(self, **kwargs):
        """(Re-)install a fake `sl` module, optionally tweaking how the fake camera behaves."""
        self.sl = make_fake_sl(**kwargs)
        for module in self._modules:
            self._monkeypatch.setattr(module, "sl", self.sl)
        self._monkeypatch.setattr(self._modules[0], "_pyzed_import_error", None)
        return self.sl

    def config(self, **kwargs):
        return self._config_cls(**{"serial_number": DEFAULT_SERIAL, "warmup_s": 0, **kwargs})

    def camera(self, **kwargs):
        camera = self._camera_cls(self.config(**kwargs))
        self._cameras.append(camera)
        return camera

    @property
    def last_camera(self):
        """The most recent fake SDK camera the backend instantiated."""
        return self.sl.cameras[-1]

    def cleanup(self):
        # Stop every read thread before monkeypatch restores the real `sl`.
        for camera in self._cameras:
            with contextlib.suppress(Exception):
                camera.disconnect()
