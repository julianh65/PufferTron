#!/usr/bin/env python3
"""
Play a Tron match against a trained PufferLib policy.

Controls:
  - Arrow keys / WASD map to global directions (up/right/down/left)
  - Turns are translated into relative Tron actions under the hood
  - Esc       quit
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib.ocean.tron.tron import Tron
from pufferlib.pufferl import load_config


ACTION_LEFT = 0
ACTION_FORWARD = 1
ACTION_RIGHT = 2

# Native Tron direction indices (sync with DIR_X/DIR_Y in tron.h)
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play against a checkpointed Tron policy trained with PufferLib."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint. Defaults to the latest model_* checkpoint under experiments/.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="experiments",
        help="Root directory to scan when auto-selecting the latest checkpoint.",
    )
    parser.add_argument(
        "--bots",
        type=int,
        default=7,
        help="Number of policy-controlled bots to spawn (total agents = bots + 1 human).",
    )
    parser.add_argument(
        "--human-index",
        type=int,
        default=0,
        help="Agent index controlled by the human player.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target frames per second for the pygame loop.",
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        default=8,
        help="Pixel scale factor used when converting the RGB array for display.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device to load the policy onto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="Environment reset seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling from the policy.",
    )
    return parser.parse_args()


def require_pygame():
    try:
        import pygame  # type: ignore
    except ImportError as exc:  # pragma: no cover - user feedback path
        print("This script requires pygame. Install it with `pip install pygame`.", file=sys.stderr)
        raise SystemExit(1) from exc
    return pygame


def find_latest_checkpoint(root: str, env_prefix: str = "puffer_tron") -> Path:
    pattern = os.path.join(root, f"{env_prefix}_*/model_{env_prefix}_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found matching {pattern!r}. "
            "Provide --checkpoint or ensure training outputs exist."
        )
    latest = max(candidates, key=os.path.getmtime)
    return Path(latest)


def build_policy(config: Dict, env: Tron, device: torch.device) -> torch.nn.Module:
    from pufferlib.ocean import torch as ocean_torch  # lazy import to avoid circulars

    policy_name = config["policy_name"]
    policy_kwargs = dict(config.get("policy", {}))
    base_cls = getattr(ocean_torch, policy_name)
    base_policy = base_cls(env, **policy_kwargs)

    rnn_name = config.get("rnn_name")
    if rnn_name:
        rnn_kwargs = dict(config.get("rnn", {}))
        rnn_cls = getattr(ocean_torch, rnn_name)
        policy = rnn_cls(env, base_policy, **rnn_kwargs)
    else:
        policy = base_policy

    policy = policy.to(device)
    policy.eval()
    return policy


def init_policy_state(policy: torch.nn.Module, batch_size: int, device: torch.device):
    if hasattr(policy, "hidden_size") and hasattr(policy, "cell"):
        hidden_size = getattr(policy, "hidden_size")
        zeros = torch.zeros(batch_size, hidden_size, device=device)
        return {"lstm_h": zeros.clone(), "lstm_c": zeros.clone()}
    return None


def policy_actions(
    policy: torch.nn.Module,
    observations: np.ndarray,
    device: torch.device,
    deterministic: bool,
    state: Optional[Dict[str, torch.Tensor]],
) -> np.ndarray:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        if state is not None:
            logits, _ = policy.forward_eval(obs_tensor, state)
        else:
            logits, _ = policy.forward_eval(obs_tensor)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions, _, _ = pufferlib.pytorch.sample_logits(logits)

    return actions.detach().to("cpu").numpy().astype(np.int32)


def desired_heading_from_keys(pygame) -> Optional[int]:
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_UP] or pressed[pygame.K_w]:
        return DIR_UP
    if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
        return DIR_RIGHT
    if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
        return DIR_DOWN
    if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
        return DIR_LEFT
    return None


def heading_from_observation(obs_row: np.ndarray, vision: int, default: int) -> int:
    orient_offset = 2 * vision * vision
    if orient_offset + 4 > obs_row.shape[0]:
        return default
    orient = obs_row[orient_offset : orient_offset + 4]
    idx = int(np.argmax(orient))
    if orient[idx] <= 0:
        return default
    return idx


def relative_action(current_heading: int, desired_heading: Optional[int]) -> int:
    if desired_heading is None:
        return ACTION_FORWARD

    delta = (desired_heading - current_heading) % 4
    if delta == 0:
        return ACTION_FORWARD
    if delta == 1:
        return ACTION_RIGHT
    if delta == 3:
        return ACTION_LEFT
    # Opposite direction (delta == 2); choose a consistent turn direction.
    return ACTION_RIGHT


def ensure_device(requested: str) -> torch.device:
    requested_device = requested.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested_device)


def main() -> None:
    args = parse_args()
    pygame = require_pygame()

    device = ensure_device(args.device)

    orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        config = load_config("puffer_tron")
    finally:
        sys.argv = orig_argv

    total_agents = args.bots + 1
    if not 0 <= args.human_index < total_agents:
        raise ValueError(f"--human-index must be between 0 and {total_agents - 1}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")

    env_kwargs = dict(config.get("env", {}))
    env_kwargs["num_agents"] = total_agents
    env_kwargs["render_scale"] = args.render_scale

    env = Tron(num_envs=1, render_mode=None, **env_kwargs)

    policy = build_policy(config, env, device)
    bot_indices = [i for i in range(total_agents) if i != args.human_index]

    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_root)
        print(f"Loaded latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loaded checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)

    policy_state = init_policy_state(policy, len(bot_indices), device)

    pygame.display.set_caption("PufferTron: Human vs Bots")
    clock = pygame.time.Clock()

    reset_seed = args.seed
    obs, _ = env.reset(seed=reset_seed)
    frame = env.rgb_array(scale=args.render_scale)
    height, width = frame.shape[:2]
    window = pygame.display.set_mode((width, height))

    vision = env._vision
    human_heading = heading_from_observation(
        obs[args.human_index], vision, default=DIR_UP
    )
    episode_rewards = np.zeros(total_agents, dtype=np.float32)
    running = True

    print(
        "Controls: arrow keys / WASD map to global directions "
        "(Esc to quit, hold key for consecutive turns)."
    )
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        desired_heading = desired_heading_from_keys(pygame)
        human_action = relative_action(human_heading, desired_heading)

        actions = np.full(total_agents, ACTION_FORWARD, dtype=np.int32)
        actions[args.human_index] = human_action

        if bot_indices:
            bot_obs = obs[bot_indices]
            bot_actions = policy_actions(
                policy,
                bot_obs,
                device=device,
                deterministic=args.deterministic,
                state=policy_state,
            )
            actions[bot_indices] = bot_actions

        obs, rewards, terminals, truncations, info = env.step(actions)
        episode_rewards += rewards
        human_heading = heading_from_observation(
            obs[args.human_index], vision, human_heading
        )

        frame = env.rgb_array(scale=args.render_scale)
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        window.blit(surface, (0, 0))
        pygame.display.flip()

        if terminals.any():
            round_rewards = episode_rewards.copy()
            human_reward = round_rewards[args.human_index]
            if bot_indices:
                bot_summary = ", ".join(
                    f"{idx}:{round_rewards[idx]:.2f}" for idx in bot_indices
                )
            else:
                bot_summary = "none"
            print(
                f"Round finished | human: {human_reward:.2f} | bot rewards -> {bot_summary}"
            )
            episode_rewards[:] = 0.0
            if policy_state is not None and bot_indices:
                policy_state = init_policy_state(policy, len(bot_indices), device)
            reset_seed += 1
            obs, _ = env.reset(seed=reset_seed)
            human_heading = heading_from_observation(
                obs[args.human_index], vision, human_heading
            )

        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
