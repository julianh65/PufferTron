from __future__ import annotations

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.tron import binding


class Tron(pufferlib.PufferEnv):

    def __init__(
        self,
        num_envs: int = 1,
        map_width: int = 64,
        map_height: int = 64,
        vision_size: int = 9,
        num_agents: int = 4,
        spawn_separation: int = 5,
        spawn_margin: int = 2,
        spawn_attempts: int = 0,
        max_round_steps: int = 256,
        reward_alive: float = 0.01,
        reward_step: float = 0.0,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_tie: float = 0.0,
        reward_timeout: float = 0.0,
        reward_kill: float = 0.0,
        log_interval: int = 256,
        render_mode: str | None = None,
        render_scale: int = 6,
        video_env_index: int = 0,
        buf=None,
        seed: int = 0,
    ):
        obs_size = vision_size * vision_size * 2 + 5
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(3)

        self.render_mode = render_mode
        self.render_scale = render_scale
        self.log_interval = log_interval
        self._vision = vision_size
        self._agents_per_env = num_agents
        self._video_env_index = video_env_index
        self.num_agents = num_envs * num_agents
        self.num_envs = num_envs
        self._map_shape = (map_height, map_width)

        self._background = np.array([12, 18, 24], dtype=np.uint8)
        self._trail_palette = np.array(
            [
                [180, 90, 40],
                [60, 120, 190],
                [98, 170, 92],
                [180, 70, 110],
                [190, 150, 60],
                [100, 90, 200],
                [230, 120, 210],
                [120, 200, 150],
                [200, 110, 70],
                [140, 140, 230],
                [240, 180, 80],
                [110, 160, 220],
            ],
            dtype=np.uint8,
        )
        self._head_palette = np.array(
            [
                [255, 160, 89],
                [122, 200, 255],
                [166, 255, 129],
                [255, 129, 162],
                [255, 214, 102],
                [156, 135, 255],
                [255, 190, 220],
                [190, 255, 210],
                [255, 180, 140],
                [205, 205, 255],
                [255, 240, 170],
                [190, 220, 255],
            ],
            dtype=np.uint8,
        )
        self._crash_color = np.array([245, 66, 86], dtype=np.uint8)
        self._crash_marker_color = np.array([255, 230, 120], dtype=np.uint8)
        self.supports_rgb_render = True

        super().__init__(buf)

        env_kwargs = dict(
            map_width=map_width,
            map_height=map_height,
            num_agents=num_agents,
            vision_size=vision_size,
            spawn_separation=spawn_separation,
            spawn_margin=spawn_margin,
            spawn_attempts=spawn_attempts,
            max_round_steps=max_round_steps,
            reward_alive=reward_alive,
            reward_step=reward_step,
            reward_win=reward_win,
            reward_loss=reward_loss,
            reward_tie=reward_tie,
            reward_timeout=reward_timeout,
            reward_kill=reward_kill,
        )

        self._c_env_handles = []
        for env_id in range(num_envs):
            start = env_id * num_agents
            end = start + num_agents
            masks_ptr = int(self.masks[start:end].ctypes.data)
            c_env = binding.env_init(
                self.observations[start:end],
                self.actions[start:end],
                self.rewards[start:end],
                self.terminals[start:end],
                self.truncations[start:end],
                seed + env_id,
                masks_ptr=masks_ptr,
                **env_kwargs,
            )
            self._c_env_handles.append(c_env)

        self.c_envs = binding.vectorize(*self._c_env_handles)
        self.tick = 0

    def reset(self, seed: int | None = None):
        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.log_interval and self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def rgb_array(self, env_index: int | None = None, scale: int | None = None) -> np.ndarray:
        """Return an RGB frame for the selected environment copy."""
        if env_index is None:
            env_index = self._video_env_index
        state = binding.env_get(self._c_env_handles[env_index])

        width = int(state["width"])
        height = int(state["height"])
        trail_owner = np.frombuffer(state["trail_owner"], dtype=np.uint8).reshape(height, width)
        head_owner = np.frombuffer(state["head_owner"], dtype=np.uint8).reshape(height, width)
        crash_map = np.frombuffer(state["crash_map"], dtype=np.uint8).reshape(height, width)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = self._background

        palette_len = len(self._trail_palette)
        for agent_idx in range(self._agents_per_env):
            agent_id = agent_idx + 1
            trail_mask = trail_owner == agent_id
            if np.any(trail_mask):
                frame[trail_mask] = self._trail_palette[agent_idx % palette_len]

        head_mask = head_owner == agent_id
        if np.any(head_mask):
            frame[head_mask] = self._head_palette[agent_idx % palette_len]

        crash_mask = crash_map > 0
        if np.any(crash_mask):
            frame[crash_mask] = self._crash_color
            head_indices = np.argwhere(head_owner > 0)
            for y, x in head_indices:
                for dy in (-1, 1):
                    ny = np.clip(y + dy, 1, height - 2)
                    frame[ny, x] = self._crash_marker_color
                for dx in (-1, 1):
                    nx = np.clip(x + dx, 1, width - 2)
                    frame[y, nx] = self._crash_marker_color

        scale = scale or self.render_scale
        if scale > 1:
            frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        return frame

    def render(self, mode: str | None = None, env_index: int | None = None):
        mode = mode or self.render_mode or "human"
        if env_index is None:
            env_index = self._video_env_index

        if mode in {"human", "raylib"}:
            binding.vec_render(self.c_envs, env_index)
            return None
        if mode == "rgb_array":
            return self.rgb_array(env_index=env_index)
        raise ValueError(f"Unsupported render mode '{mode}'")

    def close(self):
        binding.vec_close(self.c_envs)
