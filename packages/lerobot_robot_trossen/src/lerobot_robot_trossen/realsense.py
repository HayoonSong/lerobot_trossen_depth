from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

logger = logging.getLogger(__name__)
pkg_name = "pyrealsense2-macosx" if sys.platform == "darwin" else "pyrealsense2"


@CameraConfig.register_subclass("trossen_realsense_depth")
@dataclass
class TrossenRealSenseDepthConfig(CameraConfig):
    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )


class TrossenRealSenseDepth(Camera):
    """RealSense camera that keeps RGB plus color-aligned raw depth for sidecar recording."""

    def __init__(self, config: TrossenRealSenseDepthConfig):
        if rs is None:
            raise ImportError(
                f"`pyrealsense2` is required for {self.__class__.__name__}. "
                "Install the LeRobot intelrealsense extra or install pyrealsense2 in this environment."
            )
        super().__init__(config)

        self.config = config
        self.serial_number = (
            config.serial_number_or_name
            if config.serial_number_or_name.isdigit()
            else self._find_serial_number_from_name(config.serial_number_or_name)
        )
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width = self.width
        self.capture_height = self.height
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None
        self.rs_align: rs.align | None = None
        self.depth_scale: float | None = None
        self.intrinsics: dict[str, Any] | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.new_frame_event = Event()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_color_timestamp_ms: float | None = None
        self.latest_depth_timestamp_ms: float | None = None
        self.latest_capture_time_s: float | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.rs_pipeline is not None and self.rs_profile is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            found_cameras_info.append(
                {
                    "name": device.get_info(rs.camera_info.name),
                    "type": "RealSense",
                    "id": device.get_info(rs.camera_info.serial_number),
                    "firmware_version": device.get_info(rs.camera_info.firmware_version),
                    "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                    "physical_port": device.get_info(rs.camera_info.physical_port),
                    "product_id": device.get_info(rs.camera_info.product_id),
                    "product_line": device.get_info(rs.camera_info.product_line),
                }
            )

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]
        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )
        if len(found_devices) > 1:
            serial_numbers = [dev["id"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. Use a serial number. Found: {serial_numbers}"
            )
        return str(found_devices[0]["id"])

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            rs_config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
            rs_config.enable_stream(
                rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
            )
        else:
            rs_config.enable_stream(rs.stream.color)
            rs_config.enable_stream(rs.stream.depth)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras realsense` to find available cameras."
            ) from e

        self.rs_align = rs.align(rs.stream.color)
        self._configure_capture_settings()
        self._configure_depth_metadata()
        self._start_read_thread()

        if warmup:
            self.warmup_s = max(self.warmup_s, 1)
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)

        with self.frame_lock:
            if self.latest_color_frame is None or self.latest_depth_frame is None:
                raise ConnectionError(f"{self} failed to capture color/depth frames during warmup.")

        logger.info("%s connected.", self)

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        if self.fps is None:
            self.fps = stream.fps()
        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def _configure_depth_metadata(self) -> None:
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
            "K": [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        }

    def _postprocess_color(self, image: NDArray[Any]) -> NDArray[Any]:
        h, w, c = image.shape
        if c != 3:
            raise RuntimeError(f"{self} color frame channels={c} do not match expected 3.")
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} color frame shape {(h, w)} does not match {(self.capture_height, self.capture_width)}."
            )
        processed = image
        if self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)
        return processed

    def _postprocess_depth(self, depth: NDArray[Any]) -> NDArray[Any]:
        h, w = depth.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} depth frame shape {(h, w)} does not match {(self.capture_height, self.capture_width)}."
            )
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            depth = cv2.rotate(depth, self.rotation)
        return depth

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")
        if self.rs_pipeline is None or self.rs_align is None:
            raise RuntimeError(f"{self}: RealSense pipeline/align must be initialized.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                ret, frames = self.rs_pipeline.try_wait_for_frames(timeout_ms=10000)
                if not ret or frames is None:
                    raise RuntimeError(f"{self} read failed (status={ret}).")

                aligned_frames = self.rs_align.process(frames)
                color_frame_raw = aligned_frames.get_color_frame()
                depth_frame_raw = aligned_frames.get_depth_frame()
                if not color_frame_raw or not depth_frame_raw:
                    raise RuntimeError(f"{self} missing color or depth frame.")

                color_frame = self._postprocess_color(np.asanyarray(color_frame_raw.get_data()))
                depth_frame = self._postprocess_depth(np.asanyarray(depth_frame_raw.get_data()))
                capture_time_s = time.time()

                with self.frame_lock:
                    self.latest_color_frame = color_frame
                    self.latest_depth_frame = depth_frame
                    self.latest_color_timestamp_ms = float(color_frame_raw.get_timestamp())
                    self.latest_depth_timestamp_ms = float(depth_frame_raw.get_timestamp())
                    self.latest_capture_time_s = capture_time_s
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning("Error reading frame in background thread for %s: %s", self, e)
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_color_timestamp_ms = None
            self.latest_depth_timestamp_ms = None
            self.latest_capture_time_s = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")
        with self.frame_lock:
            frame = None if self.latest_color_frame is None else self.latest_color_frame.copy()
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no color frame available for {self}.")
        return frame

    def read(self) -> NDArray[Any]:
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self.frame_lock:
            frame = None if self.latest_color_frame is None else self.latest_color_frame.copy()
            timestamp = self.latest_capture_time_s
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.time() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms.")
        return frame

    @check_if_not_connected
    def get_rgbd_frame(self) -> dict[str, Any]:
        with self.frame_lock:
            if self.latest_color_frame is None or self.latest_depth_frame is None:
                raise RuntimeError(f"{self} has no RGB-D frame available.")
            return {
                "color": self.latest_color_frame.copy(),
                "depth": self.latest_depth_frame.copy(),
                "depth_scale_m_per_unit": self.depth_scale,
                "color_timestamp_ms": self.latest_color_timestamp_ms,
                "depth_timestamp_ms": self.latest_depth_timestamp_ms,
                "system_time_s": self.latest_capture_time_s,
            }

    def write_camera_info(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "serial": self.serial_number,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "aligned_depth_to": "color",
            "depth_unit": "uint16_mm",
            "depth_scale_m_per_unit": self.depth_scale,
            "intrinsics": self.intrinsics,
            "K": None if self.intrinsics is None else self.intrinsics["K"],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it appears already disconnected.")
        if self.thread is not None:
            self._stop_read_thread()
        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None
            self.rs_align = None
        logger.info("%s disconnected.", self)


def depth_to_visualization(depth: NDArray[Any], max_depth_mm: int = 2000) -> NDArray[Any]:
    depth_u8 = np.clip(depth, 0, max_depth_mm)
    depth_u8 = (depth_u8.astype(np.float32) / max_depth_mm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
