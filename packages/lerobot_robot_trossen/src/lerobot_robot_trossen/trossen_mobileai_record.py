import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, make_default_processors
from lerobot.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig as _LeRobotRecordConfig,  # noqa: F401
)
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, sanity_check_dataset_name
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trossen import MobileAIRobotConfig  # noqa: F401
from lerobot_robot_trossen.config_mobileai import MobileAIRobotConfig as _MobileAIRobotConfig  # noqa: F401
from lerobot_robot_trossen.realsense import (
    TrossenRealSenseDepthConfig,  # noqa: F401
    depth_to_visualization,
)
from lerobot_teleoperator_trossen import MobileAILeaderTeleopConfig  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class RgbdConfig:
    enabled: bool = False
    save_color: bool = True
    save_depth: bool = True
    save_depth_vis: bool = True
    max_depth_mm: int = 2000


@dataclass
class TrossenMobileAIRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    rgbd: RgbdConfig = field(default_factory=RgbdConfig)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError("A teleoperator is required for recording. Use --teleop.type=... to specify one.")


class RgbdSidecarWriter:
    def __init__(self, root: Path, cfg: RgbdConfig):
        self.root = root / "rgbd"
        self.cfg = cfg
        self.rows: list[dict[str, Any]] = []

    def initialize(self, robot: Robot) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for cam_key, cam in getattr(robot, "cameras", {}).items():
            if hasattr(cam, "write_camera_info"):
                cam.write_camera_info(self.root / cam_key / "camera.json")

    def write_frame(self, robot: Robot, dataset: LeRobotDataset) -> None:
        if not self.cfg.enabled or dataset.writer is None:
            return

        episode_index = int(dataset.writer.episode_buffer["episode_index"])
        frame_index = int(dataset.writer.episode_buffer["size"])

        for cam_key, cam in getattr(robot, "cameras", {}).items():
            if not hasattr(cam, "get_rgbd_frame"):
                continue

            rgbd_frame = cam.get_rgbd_frame()
            episode_dir = self.root / cam_key / f"episode_{episode_index:06d}"
            color_dir = episode_dir / "color"
            depth_dir = episode_dir / "depth"
            depth_vis_dir = episode_dir / "depth_vis"
            stem = f"{frame_index:06d}.png"

            row = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "camera": cam_key,
                "color_path": "",
                "depth_path": "",
                "depth_vis_path": "",
                "system_time_s": rgbd_frame["system_time_s"],
                "color_timestamp_ms": rgbd_frame["color_timestamp_ms"],
                "depth_timestamp_ms": rgbd_frame["depth_timestamp_ms"],
            }

            if self.cfg.save_color:
                color_dir.mkdir(parents=True, exist_ok=True)
                color_path = color_dir / stem
                cv2.imwrite(str(color_path), cv2.cvtColor(rgbd_frame["color"], cv2.COLOR_RGB2BGR))
                row["color_path"] = str(color_path.relative_to(self.root))

            if self.cfg.save_depth:
                depth_dir.mkdir(parents=True, exist_ok=True)
                depth_path = depth_dir / stem
                cv2.imwrite(str(depth_path), rgbd_frame["depth"])
                row["depth_path"] = str(depth_path.relative_to(self.root))

            if self.cfg.save_depth_vis:
                depth_vis_dir.mkdir(parents=True, exist_ok=True)
                depth_vis_path = depth_vis_dir / stem
                cv2.imwrite(str(depth_vis_path), depth_to_visualization(rgbd_frame["depth"], self.cfg.max_depth_mm))
                row["depth_vis_path"] = str(depth_vis_path.relative_to(self.root))

            self.rows.append(row)

    def clear_episode(self, episode_index: int) -> None:
        if not self.cfg.enabled:
            return
        self.rows = [row for row in self.rows if row["episode_index"] != episode_index]
        for cam_dir in self.root.iterdir() if self.root.exists() else []:
            ep_dir = cam_dir / f"episode_{episode_index:06d}"
            if ep_dir.exists():
                for child in ep_dir.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(ep_dir.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                ep_dir.rmdir()

    def finalize(self) -> None:
        if not self.cfg.enabled:
            return
        if not self.rows:
            return
        csv_path = self.root / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


def _episode_buffer_size(dataset: LeRobotDataset) -> int:
    writer = getattr(dataset, "writer", None)
    episode_buffer = getattr(writer, "episode_buffer", None) if writer is not None else None
    if not episode_buffer:
        return 0
    return int(episode_buffer.get("size", 0))


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    control_time_s: int | float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    rgbd_writer: RgbdSidecarWriter | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    control_interval = 1 / fps
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if teleop is None:
            logging.warning("No teleoperator provided, skipping action generation.")
            continue

        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs))
        action_values = act_processed_teleop
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            if rgbd_writer is not None:
                rgbd_writer.write_frame(robot, dataset)
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values, compress_images=display_compressed_images)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                "Record loop is running slower (%.1f Hz) than the target FPS (%s Hz).",
                1 / dt_s,
                fps,
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: TrossenMobileAIRecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    if teleop_action_processor is None or robot_action_processor is None or robot_observation_processor is None:
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    sidecar = None
    listener = None
    try:
        if cfg.resume:
            raise NotImplementedError("RGB-D sidecar recording currently requires --resume=false.")

        repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
        if repo_name.startswith("eval_"):
            raise ValueError("Dataset names starting with 'eval_' are reserved for policy evaluation.")
        sanity_check_dataset_name(cfg.dataset.repo_id, None)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

        sidecar = RgbdSidecarWriter(dataset.root, cfg.rgbd)

        robot.connect()
        if teleop is not None:
            teleop.connect()
        sidecar.initialize(robot)

        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    rgbd_writer=sidecar,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    sidecar.clear_episode(dataset.num_episodes)
                    dataset.clear_episode_buffer()
                    continue

                episode_size = _episode_buffer_size(dataset)
                if episode_size == 0:
                    sidecar.clear_episode(dataset.num_episodes)
                    dataset.clear_episode_buffer()
                    raise RuntimeError(
                        "No frames were recorded for the current episode. "
                        "This usually means the episode was exited before the first frame was added, "
                        "or the recording loop never reached dataset.add_frame()."
                    )

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if sidecar is not None:
            sidecar.finalize()
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        try:
            if not is_headless() and listener:
                listener.stop()
        except Exception:
            pass
        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved, skipping push to hub.")
        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
