import json
import sys
import jsonlines
import os
import shutil
import time
import warnings
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI
from scipy.spatial import cKDTree
import cv2
import numpy as np

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import (
    Open_Nav,
    MAX_HISTORY_STEPS,
    NAVIGATOR_MAX_TOKENS,
    NAVIGATOR_NUM_OUTPUT,
)
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *
from vlnce_baselines.common.map import get_structure_wp

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

try:
    # This repo doesn't use tensorflow for Open-Nav inference, but the original
    # upstream code imports it. Make it an optional dependency so runs won't
    # fail in environments without tensorflow installed.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import tensorflow as tf  # noqa: F401
except ImportError:
    tf = None

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        # Select compute device (GPU if available, otherwise CPU).
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
        else:
            print("没有GPU,使用CPU", flush=True)
            self.device = torch.device("cpu")
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                './waypoint_prediction/checkpoints/check_val_best_avg_wayscore',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index]
            }
            
        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        waypoint_ids = {}
        # -1 不是“取最后一个环境”，而是“取最后一次/最终版本的角度与距离输出
        angles = batch_angles[-1] 
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
                waypoint_ids['1'] = "0 - 30°"
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
                waypoint_ids['2'] = "30 - 60°"
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
                waypoint_ids['3'] = "60 - 90°"
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
                waypoint_ids['4'] = "90 - 120°"
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
                waypoint_ids['5'] = "120 - 150°"
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
                waypoint_ids['6'] = "150 - 180°"
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
                waypoint_ids['7'] = "180 - 210°"
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
                waypoint_ids['8'] = "210 - 240°"
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
                waypoint_ids['9'] = "240 - 270°"
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
                waypoint_ids['10'] = "270 - 300°"
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
                waypoint_ids['11'] = "300 - 330°"
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                waypoint_ids['0'] = "330 - 360°"
        return waypoint_images, waypoint_radius, waypoint_distances, waypoint_ids

    def safe_remove_keys(self, original_dict, keys_to_remove):
        """Remove keys in `keys_to_remove` from a dict, returning (remaining, removed)."""
        removed_items = {k: v for k, v in original_dict.items() if k in keys_to_remove}
        modified_dict = {k: v for k, v in original_dict.items() if k not in keys_to_remove}
        if modified_dict:
            return modified_dict, removed_items
        return original_dict, {}

    def compute_absolute_positions(self, pos, heading, angle_dict, distance_dict):
        """Convert candidate (relative angle, distance) pairs to global 3D coordinates."""
        result = {}
        for pid in angle_dict:
            rel_angle = angle_dict[pid]
            distance = distance_dict[pid]
            global_angle = (heading + rel_angle) % (2 * np.pi)
            x = pos[0] - distance * np.sin(global_angle)
            y = pos[1]
            z = pos[2] - distance * np.cos(global_angle)
            result[pid] = (x, y, z)
        return result

    def find_candidates_on_path(self, candidate_points_dict, path_points, threshold=0.1):
        """Match candidate points that lie close to previously visited path points."""
        ids = list(candidate_points_dict.keys())
        if not ids:
            return []
        candidate_coords = np.array([candidate_points_dict[i] for i in ids])
        if len(path_points) == 0:
            return []
        tree = cKDTree(path_points)
        dists, _ = tree.query(candidate_coords, k=1)
        matched_ids = [ids[i] for i in range(len(dists)) if dists[i] < threshold]
        return matched_ids

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) numpy array
        DATASET = "R2R"
        if DATASET == "R2R":
            min_depth = 0.0
            max_depth = 10.0
        elif DATASET == "RxR":
            min_depth = 0.5
            max_depth = 5.0

        depth = depth * 1.0
        H = depth.shape[1]
        depth_max = np.max(depth, axis=1, keepdims=True)  # (B, 1, W, 1)
        depth_max = np.tile(depth_max, (1, H, 1, 1))
        depth[depth == 0] = depth_max[depth == 0]

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.0
        return depth[:, :, :, 0]

    def image_get_rel_position(self, depth_map, angle, shape=(112, 112)):
        DATASET = "R2R"
        W = shape[0]
        H = shape[0]
        half_W = W // 2
        half_H = H // 2
        depth_y = depth_map.astype(np.float32)

        if DATASET == "R2R":
            tan_xy = (
                np.array(([i / half_W + 1 / W for i in range(-half_W, half_W)]) * H, np.float32)
                * math.tan(math.pi / 4)
            )
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (
                np.array([[i / half_H - 1 / H for i in range(half_H, -half_H, -1)]] * W, np.float32).T.reshape(
                    (-1,)
                )
                * math.tan(math.pi / 4.0)
            )
        elif DATASET == "RxR":
            tan_xy = (
                np.array(([i / half_W + 1 / W for i in range(-half_W, half_W)]) * H, np.float32)
                * math.tan(math.pi * 79.0 / 360.0)
            )
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (
                np.array([[i / half_H - 1 / H for i in range(half_H, -half_H, -1)]] * W, np.float32).T.reshape(
                    (-1,)
                )
                * math.tan(math.pi * 79.0 / 360.0)
            )

        direction = (direction + angle) % (2 * math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z
        return rel_x, rel_z, rel_y, direction.reshape(-1)

    def getGlobalMap(self, position, heading, depths, shape=(112, 112)):
        """Build global point cloud from multi-view depth maps."""
        depth = [cv2.resize(obs, shape, interpolation=cv2.INTER_NEAREST) for obs in depths]
        depth = [depth[0]] + depth[1:][::-1]
        depth = np.stack(depth, 0).reshape([len(depths), shape[0], shape[1], 1])
        depth = self.preprocess_depth(depth)

        pcd_x = []
        pcd_y = []
        pcd_z = []
        for ix in range(len(depth)):
            dep = depth[ix : ix + 1].reshape(-1)
            rel_x, rel_y, rel_z, direction = self.image_get_rel_position(
                dep, ix * math.pi / (len(depths) / 2)
            )
            rel_x = rel_x[dep < 5]
            rel_y = rel_y[dep < 5]
            rel_z = rel_z[dep < 5]
            pcd_x.append(rel_x)
            pcd_y.append(rel_y)
            pcd_z.append(rel_z)

        pcd_x = np.concatenate(pcd_x, axis=-1)
        pcd_y = np.concatenate(pcd_y, axis=-1)
        pcd_z = np.concatenate(pcd_z, axis=-1)
        pcd = np.stack([pcd_x, pcd_y, pcd_z], -1)
        return pcd

    def _world_to_occ_pixel(self, world_x, world_z, agent_pos, heading, map_size, voxel_size=0.03):
        """Project Habitat world (x,z) to pixels consistent with points_to_occ_map_centered.

        Occupancy is built from depth in an agent-centered frame; candidates use
        compute_absolute_positions (global x,z from heading + polar). Inverse:
        global_angle = atan2(-dx, -dz), rel_angle = global_angle - heading,
        local_right = -d*sin(rel_angle), local_forward = d*cos(rel_angle).
        """
        h, w = map_size
        cx, cy = w // 2, h // 2
        dx = float(world_x) - float(agent_pos[0])
        dz = float(world_z) - float(agent_pos[2])
        ga = math.atan2(-dx, -dz)
        rel_angle = ga - float(heading)
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        d = math.hypot(dx, dz)
        lx = -d * math.sin(rel_angle)
        lz = d * math.cos(rel_angle)
        px = int(round(lx / voxel_size)) + cx
        py = int(round(lz / voxel_size)) + cy
        return px, py

    def save_step_occupancy_map(
        self,
        navi_area,
        cand_pos,
        selected_vp,
        vis_positions,
        map_size,
        save_path,
        voxel_size=0.03,
        agent_pos=None,
        heading=None,
    ):
        """
        Save occupancy map visualization for one step:
        - history trajectory points (green)
        - candidate waypoints (blue)
        - selected waypoint (red)
        """
        if navi_area is None:
            return
        if agent_pos is None or heading is None:
            return

        if navi_area.ndim == 2:
            canvas = cv2.cvtColor(navi_area.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            canvas = navi_area.copy().astype(np.uint8)

        # History trajectory points
        for p in vis_positions:
            if p is None or len(p) < 3:
                continue
            px, py = self._world_to_occ_pixel(
                float(p[0]), float(p[2]), agent_pos, heading, map_size, voxel_size=voxel_size
            )
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), 2, (0, 220, 0), thickness=-1)  # green

        # Candidate waypoints
        for _, p in cand_pos.items():
            if p is None or len(p) < 3:
                continue
            px, py = self._world_to_occ_pixel(
                float(p[0]), float(p[2]), agent_pos, heading, map_size, voxel_size=voxel_size
            )
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), 4, (255, 90, 0), thickness=-1)  # blue-ish

        # Selected waypoint
        selected_key = str(selected_vp) if selected_vp is not None else None
        if selected_key is not None and selected_key in cand_pos:
            p = cand_pos[selected_key]
            if p is not None and len(p) >= 3:
                px, py = self._world_to_occ_pixel(
                    float(p[0]), float(p[2]), agent_pos, heading, map_size, voxel_size=voxel_size
                )
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), thickness=-1)  # red

        cv2.imwrite(save_path, canvas)

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        # TOP_DOWN_MAP_VLNCE needs `data/connectivity_graphs.pkl` (MP3D connectivity graphs).
        # If the file is missing, skip this measurement and use a fixed map_size for SWG (see loop below).
        graphs_file = config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE
        graphs_path = (
            graphs_file if os.path.isabs(graphs_file) else os.path.join(os.getcwd(), graphs_file)
        )
        if os.path.exists(graphs_path):
            if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        else:
            # Drop measure if merged config or other code already added it.
            ms = [m for m in list(config.TASK_CONFIG.TASK.MEASUREMENTS) if m != "TOP_DOWN_MAP_VLNCE"]
            config.TASK_CONFIG.TASK.MEASUREMENTS = ms
            print(
                f"[Open-Nav] Missing connectivity graphs file: {graphs_path}\n"
                "  Skipping TOP_DOWN_MAP_VLNCE (SWG will use default map_size from get_structure_wp).\n"
                "  To enable the full top-down map measure, place connectivity_graphs.pkl or set "
                "TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE to the correct path.",
                flush=True,
            )
        if "COLLISIONS" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = input("Do you want to overwrite the results? (yes/no): ").strip().lower()
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
                

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        #envs.number_of_episodes = [1] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval() 
        self.waypoint_predictor.eval()
        observations = envs.reset()

        # 打印observations的类型、长度、keys和instruction
        # print("observations的类型:", type(observations), flush=True)
        # print()
        # print("observations的长度:", len(observations), flush=True)
        # print()
        # print("observations的keys:", observations[0].keys(), flush=True)
        # print()
        # print("observations的instruction:", observations[0]["instruction"], flush=True)
        # print()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        # 判断评估剧集的数量是否为-1，如果为-1，则评估所有剧集
        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            # 如果运行剧集的数量不为-1，则运行指定数量的剧集
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # Save selected next_vp RGB images under one folder per episode.
        selected_vp_rgb_root = "./selected_next_vp_rgb"
        os.makedirs(selected_vp_rgb_root, exist_ok=True)
        active_save_episode_id = None
        active_save_episode_dir = None

        # 设置日志记录器
        # set up the logger
        log_file = "./navigator_log.log"
        # Start each evaluation run with a clean navigator log file.
        if os.path.exists(log_file):
            os.remove(log_file)
        import logging
        logging.basicConfig(
            # Log format includes timestamp, filename/function, line number and level.
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            # Human-readable time format.
            datefmt="%Y-%m-%d %H:%M:%S",
            # Allow overriding the log level via env var (default: INFO).
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            # Also send logs to stdout so you can see them live in the terminal.
            stream=sys.stdout,
            # Append mode: keep writing to the same log file.
            filemode="a"
        )
        # Create/retrieve a named logger instance so navigator code can call nav_logger.info(...)
        nav_logger = logging.getLogger("vln_logger")
        # Ensure logs are written into navigator_log.log.
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        

        # Initialize the dataset name.
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache.json"
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {} 

        
        # Initialize the navigator.
        navigator = Open_Nav(self.device, llm_type=config.LLM, api_key=config.API_KEY)
        # Initialize the current step and navigation history.
        current_step = 0
        nav_history = []
        # Initialize the error number.
        error_number = 0
        # SWG visited-path cache (used to filter already-seen candidate viewpoints).
        vis_positions = []

        # Start the evaluation loop.
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            try:
                current_episodes = envs.current_episodes()
            except Exception:
                nav_logger.exception(
                    "VectorEnv worker failed when fetching current episodes. "
                    "This usually means a previous worker-side error (OOM/crash) "
                    "or an unfinished env pipe read/write."
                )
                break
            current_episode_id = str(current_episodes[0].episode_id)

            # Save the selected next_vp RGB images under one folder per episode.
            if current_episode_id != active_save_episode_id:
                active_save_episode_id = current_episode_id
                active_save_episode_dir = os.path.join(
                    selected_vp_rgb_root, f"episode_{active_save_episode_id}"
                )
                # If this episode folder already exists from a previous run, reset it.
                if os.path.exists(active_save_episode_dir):
                    shutil.rmtree(active_save_episode_dir)
                os.makedirs(active_save_episode_dir, exist_ok=True)
                nav_logger.info(
                    f">>> Prepared RGB save directory for episode: {active_save_episode_dir}"
                )
                # Save the episode instruction as text alongside step RGB images.
                instruction_txt_path = os.path.join(active_save_episode_dir, "instruction.txt")
                with open(instruction_txt_path, "w", encoding="utf-8") as f_instruction:
                    f_instruction.write(str(instruction).strip() + "\n")
                nav_logger.info(f">>> Saved episode instruction text: {instruction_txt_path}")
                # Reset visited-path cache for each episode.
                vis_positions = []


            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
                vis_positions.append(agent_state_i["position"])
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info(">>> Instruction: "+instruction)
            actions, landmarks = "", ""
            if instruction not in actions_cache.keys():
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(instruction)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
            nav_logger.info(">>> Actions: "+ actions + "\n")
            nav_logger.info(">>> Landmarks: " + landmarks + "\n")
            
            # step_length = 6 if len(actions.split("\n")) <= 6 else 8 
            # 分解动作数量的两倍作为最大执行步数，如果动作数量小于5，则执行7步，如果动作数量大于5，则执行9步
            step_length = 7 if len(actions.split("\n")) <= 5 else 9 


            stop_flag = False
            current_step += 1

            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            nav_logger.info("========== Get waypoint ids ==========")

            # ========== SWG candidate generation (llm2 logic) ==========
            info = envs.get_metrics()
            td = (
                info[0].get("top_down_map_vlnce")
                if info and len(info) > 0
                else None
            )
            if td is not None and isinstance(td, dict) and td.get("map") is not None:
                map_size = td["map"].shape
            else:
                # Same default as vlnce_baselines.common.map.get_structure_wp when no Habitat top-down map.
                map_size = (1024, 1736)
                nav_logger.info(
                    f">>> SWG: top_down_map_vlnce not in metrics; using default map_size={map_size}"
                )
            depths = [observations[0]["depth"][:, :, 0]] + [
                v[:, :, 0] for k, v in observations[0].items() if "depth_" in k
            ]
            pcd = self.getGlobalMap(positions[0], headings[0], depths)
            wp_radius, wp_distance, navi_area = get_structure_wp(
                pcd,
                clamp_dist=(1, 1.5),
                map_size=map_size,
            )
            # llm.py's construct_image_dicts expects batch_angles to be a sequence.
            images_dict, radius_dict, distance_dict, ids_dict = self.construct_image_dicts(
                wp_distance,
                [wp_radius],
                images_list,
            )

            # visited-path filtering (remove candidates already near previous visited positions).
            cand_pos = self.compute_absolute_positions(
                positions[0], headings[0], radius_dict, distance_dict
            )
            matched = self.find_candidates_on_path(
                cand_pos, np.array(vis_positions[:-1]), threshold=0.5
            )
            images_dict, _ = self.safe_remove_keys(images_dict, matched)
            radius_dict, _ = self.safe_remove_keys(radius_dict, matched)
            distance_dict, _ = self.safe_remove_keys(distance_dict, matched)
            ids_dict, _ = self.safe_remove_keys(ids_dict, matched)
            cand_pos, _ = self.safe_remove_keys(cand_pos, matched)

            if current_step == 1:
                nav_logger.info(
                    "[DEBUG key check] "
                    f"images_dict keys={sorted(list(images_dict.keys()))}, "
                    f"radius_dict keys={sorted(list(radius_dict.keys()))}, "
                    f"distance_dict keys={sorted(list(distance_dict.keys()))}"
                )

            nav_logger.info(f">>> Waypoint ids:\n{ids_dict}\n")

            nav_logger.info("========== Get Observation ==========")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # free waypoint predictor cache before SpatialBot/RAM
            observation, observe_dict = navigator.observe_environment(nav_logger, current_step, images_dict)
            
            nav_logger.info("========== Review History ==========")
            history_traj = navigator.review_history(nav_logger, nav_history, last_k_steps=MAX_HISTORY_STEPS) if len(nav_history) > 0 else "Step 0 start position. "

            try:
                if not stop_flag:
                    nav_logger.info("========== Estimate Completion Progress ==========")
                    estimation = navigator.estimate_completion(nav_logger, actions, landmarks, history_traj, nav_history=nav_history)

                    nav_logger.info("========== Next Action Prediction ==========")
                    predictions, thoughts, break_flag = navigator.move_to_next_vp(
                        nav_logger, current_step, instruction, actions, landmarks, history_traj, estimation, observation, observe_dict,
                        num_output=NAVIGATOR_NUM_OUTPUT,
                        max_tokens=NAVIGATOR_MAX_TOKENS,
                    )

                    nav_logger.info("========== Thought ==========")
                    fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)

                    nav_logger.info("========== Test Decision ==========")
                    next_vp, thought, error_number = navigator.test_decisions(nav_logger, fused_pred_thought, observation, instruction, error_number, observe_dict)

                    env_actions = []
                    vp_key = str(next_vp)
                    if vp_key not in radius_dict or vp_key not in distance_dict:
                        # Safety fallback: keep the run from crashing on rare key mismatches.
                        common_keys = [
                            k for k in radius_dict.keys()
                            if k in distance_dict and k in observe_dict
                        ]
                        if not common_keys:
                            nav_logger.info(
                                f"[WARN] next_vp key mismatch: vp_key={vp_key} "
                                f"radius_keys={list(radius_dict.keys())} distance_keys={list(distance_dict.keys())}. "
                                "Cannot find fallback candidate."
                            )
                            raise KeyError(vp_key)
                        nav_logger.info(
                            f"[WARN] next_vp key mismatch: vp_key={vp_key}; fallback to {common_keys[0]}"
                        )
                        vp_key = common_keys[0]
                        next_vp = vp_key

                    # Save per-step occupancy map with history/candidates/selected waypoint.
                    occupancy_save_name = f"step_{current_step:03d}_occupancy.png"
                    occupancy_save_path = os.path.join(active_save_episode_dir, occupancy_save_name)
                    self.save_step_occupancy_map(
                        navi_area=navi_area,
                        cand_pos=cand_pos,
                        selected_vp=vp_key,
                        vis_positions=vis_positions,
                        map_size=map_size,
                        save_path=occupancy_save_path,
                        voxel_size=0.03,
                        agent_pos=positions[0],
                        heading=headings[0],
                    )
                    nav_logger.info(
                        f">>> Saved occupancy map (green=history, blue=candidates, red=selected): "
                        f"{occupancy_save_path}"
                    )
                    # Save the RGB image corresponding to the selected next_vp.
                    # Filename includes episode id, step id and selected viewpoint id.
                    if vp_key in images_dict and "rgb" in images_dict[vp_key]:
                        rgb_save_name = f"step_{current_step:03d}_vp_{vp_key}.jpg"
                        rgb_save_path = os.path.join(active_save_episode_dir, rgb_save_name)
                        images_dict[vp_key]["rgb"].save(rgb_save_path, format="JPEG")
                        nav_logger.info(f">>> Saved selected next_vp RGB image: {rgb_save_path}")
                    else:
                        nav_logger.info(
                            f">>> Skip saving RGB: vp_key={vp_key} not found in images_dict or missing rgb."
                        )
                    env_actions.append({'action':
                        {'action': 4,
                        'action_args':{
                            'angle': radius_dict[vp_key],
                            'distance': distance_dict[vp_key],
                        }}})
                    nav_logger.info(f">>> Selected next viewpoint ID: {next_vp}\n")
                    nav_logger.info(f">>> The final env action: {env_actions}\n")
                    outputs = envs.step(env_actions)
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                    # Save a composed visualization (rgb/depth/overhead_rgb + top-down + history points).
                    try:
                        frame = observations_to_image(
                            observations[0],
                            infos[0],
                            history_positions=vis_positions,
                        )
                        mosaic_save_name = f"step_{current_step:03d}_mosaic.jpg"
                        mosaic_save_path = os.path.join(
                            active_save_episode_dir,
                            mosaic_save_name,
                        )
                        cv2.imwrite(
                            mosaic_save_path,
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )
                        nav_logger.info(f">>> Saved composed nav frame: {mosaic_save_path}")
                    except Exception as vis_exc:
                        nav_logger.info(f"[WARN] Failed to save composed nav frame: {vis_exc}")

                    curr_observe = observe_dict[vp_key]
                    nav_logger.info("========== save history ==========")
                    nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)

                    instruction, images_list = self.generate_input(observations[-1])
                    error_number = 0 
                    # finish navigation
                    if current_step == step_length:
                        dones[0] = True 
                    else:
                        for j, ob in enumerate(observations):
                            envs.call_at(j, 
                                'change_current_path',
                                {'new_path': ob.pop('positions'),
                                'collisions': ob.pop('collisions')}
                            )
                # stop_flag is always False in this trainer; no alternate branch.

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    current_step = 0
                    nav_history = []
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    ep_info = envs.current_episodes()[i].info
                    if 'current_path' in ep_info.keys():
                        positions_ = np.array(ep_info['current_path']).astype(float)
                        collisions_ = np.array(ep_info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                        n_seg = max(0, positions_.shape[0] - 1)
                        if 'collisions' in ep_info and len(ep_info['collisions']) == n_seg:
                            collisions_ = np.array(ep_info['collisions'], dtype=float)
                        else:
                            collisions_ = np.zeros(n_seg, dtype=float)
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    metric['collisions'] = (
                        float(collisions_.mean()) if collisions_.size > 0 else 0.0
                    )
                    gt_length = distance[0]
                    denom = max(float(gt_length), float(metric['path_length']), 1e-8)
                    metric['spl'] = metric['success'] * float(gt_length) / denom

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW
                    stats_episodes[current_episodes[i].episode_id] = metric 

                    nav_logger.info(
                        ">>> Episode finished; calling reset_at (Habitat will tear down the "
                        "current scene and load the next episode — deconstruct/init logs are normal)."
                    )
                    try:
                        observations[i] = envs.reset_at(i)[0]
                    except EOFError as eof_exc:
                        # Worker died while loading the next scene; IPC read returns EOF.
                        raise RuntimeError(
                            "Habitat VectorEnv worker process died during reset_at (next episode). "
                            "Typical causes: Habitat-Sim GPU OOM or native crash during scene "
                            "teardown/load, Linux OOM killer, or too-small Docker /dev/shm. "
                            "Check: dmesg for OOM/kill, nvidia-smi, worker stderr; try "
                            "--shm-size=8g or larger for the container."
                        ) from eof_exc
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()

            except Exception as exc:
                # str(exc) can be empty (e.g. RuntimeError() from native code); log type + repr.
                nav_logger.exception(
                    "Unhandled exception in _eval_llm loop (%s: %r). "
                    "Stopping evaluation to avoid corrupted VectorEnv state.",
                    type(exc).__name__,
                    exc,
                )
                break
        try:
            envs.close()
        except Exception:
            logger.warning(
                "envs.close() failed (worker process may already be dead); ignoring.",
                exc_info=True,
            )
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        if num_episodes == 0:
            logger.warning(
                "No episodes completed successfully (stats_episodes is empty). "
                "Skipping aggregate metrics; check earlier errors in the eval loop."
            )
        else:
            for stat_key in next(iter(stats_episodes.values())).keys():
                aggregated_stats[stat_key] = (
                    sum(v[stat_key] for v in stats_episodes.values())
                    / num_episodes
                )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        return trajectories
        
    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()

