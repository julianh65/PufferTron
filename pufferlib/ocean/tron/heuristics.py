import collections
import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import pufferlib
from pufferlib.ocean.tron.tron import Tron

ACTION_LEFT = 0
ACTION_FORWARD = 1
ACTION_RIGHT = 2


def _parse_orientation(obs_tail: np.ndarray) -> int:
    """Return orientation index from the final five floats in the observation."""
    if obs_tail.shape[0] < 4:
        return 0
    return int(np.argmax(obs_tail[:4]))


def _rotate_view(view: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate a local view so that the agent's forward direction points up."""
    if orientation % 4 == 0:
        return view
    return np.rot90(view, k=-orientation)


@dataclass
class HeuristicResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    return_total: float = 0.0

    def totals(self) -> Dict[str, float]:
        total_games = self.wins + self.draws + self.losses
        total_games = max(total_games, 1)
        return {
            "win_rate": self.wins / total_games,
            "draws": float(self.draws),
        }


class TronHeuristic:
    """Base class for simple scripted heuristics."""

    def __init__(self, vision: int):
        self.vision = vision
        self.center = vision // 2

    def reset(self):
        """Reset any per-episode state."""

    def act(self, observation: np.ndarray) -> int:
        raise NotImplementedError

    def _prepare_view(self, observation: np.ndarray) -> Tuple[np.ndarray, int]:
        window_elems = self.vision * self.vision * 2
        window = observation[:window_elems].reshape(self.vision, self.vision, 2)
        occupancy = window[..., 0]
        orientation = _parse_orientation(observation[-5:])
        oriented = _rotate_view(occupancy, orientation)
        oriented = oriented.copy()
        oriented[self.center, self.center] = 1.0
        return oriented, orientation


class ForwardHeuristic(TronHeuristic):
    """Drive straight; choose the clearer side when blocked."""

    def act(self, observation: np.ndarray) -> int:
        view, _ = self._prepare_view(observation)
        c = self.center
        ahead = view[c - 1, c]
        left = view[c, c - 1]
        right = view[c, c + 1]

        if ahead < 0.5:
            return ACTION_FORWARD
        if left >= 0.5 and right < 0.5:
            return ACTION_RIGHT
        if right >= 0.5 and left < 0.5:
            return ACTION_LEFT
        # Both sides available; bias to the side with more immediate space
        return ACTION_LEFT if left <= right else ACTION_RIGHT


class WallHuggerHeuristic(TronHeuristic):
    """Follow the right-hand wall where possible."""

    def act(self, observation: np.ndarray) -> int:
        view, _ = self._prepare_view(observation)
        c = self.center
        ahead = view[c - 1, c]
        left = view[c, c - 1]
        right = view[c, c + 1]

        if right < 0.5:
            return ACTION_RIGHT
        if ahead < 0.5:
            return ACTION_FORWARD
        if left < 0.5:
            return ACTION_LEFT
        return ACTION_RIGHT


class GreedySpaceHeuristic(TronHeuristic):
    """Choose the turn that maximizes nearby reachable space."""

    def __init__(self, vision: int, max_depth: Optional[int] = None):
        super().__init__(vision)
        self.max_depth = max_depth or vision

    def _flood_fill(self, grid: np.ndarray, start: Tuple[int, int]) -> int:
        y, x = start
        if not (0 <= y < self.vision and 0 <= x < self.vision):
            return 0
        if grid[y, x] >= 0.5:
            return 0

        visited = set()
        queue = collections.deque()
        queue.append((y, x, 0))
        visited.add((y, x))
        reachable = 0

        while queue:
            cy, cx, dist = queue.popleft()
            reachable += 1
            if dist >= self.max_depth:
                continue

            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < self.vision and 0 <= nx < self.vision):
                    continue
                if grid[ny, nx] >= 0.5:
                    continue
                if (ny, nx) in visited:
                    continue
                visited.add((ny, nx))
                queue.append((ny, nx, dist + 1))

        return reachable

    def act(self, observation: np.ndarray) -> int:
        view, _ = self._prepare_view(observation)
        c = self.center

        candidates = {
            ACTION_LEFT: (c, c - 1),
            ACTION_FORWARD: (c - 1, c),
            ACTION_RIGHT: (c, c + 1),
        }
        priorities = {ACTION_FORWARD: 2, ACTION_LEFT: 1, ACTION_RIGHT: 0}

        scores = {}
        for action, start in candidates.items():
            scores[action] = self._flood_fill(view, start)

        best_action = max(scores.items(), key=lambda item: (item[1], priorities[item[0]]))[0]
        return best_action


class TronHeuristicEvaluator:
    """Runs the current policy against scripted heuristics for quick diagnostics."""

    def __init__(self, config: Dict):
        env_args = copy.deepcopy(config.get("env_args", {}))
        self.device = config["device"]
        self.episodes = int(config.get("heuristic_eval_episodes", 5) or 5)
        self.num_agents = int(config.get("heuristic_eval_num_agents", 2) or 2)
        if self.num_agents < 2:
            self.num_agents = 2
        self.num_envs = int(config.get("heuristic_eval_num_envs", 1) or 1)
        self.base_seed = int(config.get("seed", 0) or 0)

        self.use_rnn = bool(config.get("use_rnn", False))
        self.env_args = env_args
        self.env_args["num_envs"] = self.num_envs
        self.env_args["num_agents"] = self.num_agents
        self.env_args.setdefault("log_interval", 0)
        self.env_args.setdefault("render_mode", None)

        vision = env_args.get("vision_size", 9)
        self.heuristics: Dict[str, TronHeuristic] = {
            "forward": ForwardHeuristic(vision),
            "wall_hugger": WallHuggerHeuristic(vision),
            "greedy": GreedySpaceHeuristic(vision),
        }

    def run(self, policy: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        results = {}

        policy_model = getattr(policy, "module", policy)
        training_state = policy_model.training
        policy.eval()
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        try:
            for name, heuristic in self.heuristics.items():
                stats = self._evaluate(policy, heuristic, name)
                results[f"{name}"] = stats.totals()
        finally:
            torch.set_grad_enabled(prev_grad)
            if training_state:
                policy.train()

        return results

    def _evaluate(self, policy: torch.nn.Module, opponent: TronHeuristic, name: str) -> HeuristicResult:
        env = Tron(**self.env_args)
        result = HeuristicResult()
        policy_indices = [0]
        opponent_indices = [i for i in range(self.num_agents) if i not in policy_indices]

        hidden_size = getattr(getattr(policy, "module", policy), "hidden_size", None)

        try:
            for episode in range(self.episodes):
                opponent.reset()
                obs, _ = env.reset(seed=self.base_seed + episode)
                episode_returns = np.zeros(self.num_agents, dtype=np.float32)

                lstm_state = None
                if self.use_rnn and hidden_size is not None:
                    lstm_state = dict(
                        lstm_h=torch.zeros(len(policy_indices), hidden_size, device=self.device),
                        lstm_c=torch.zeros(len(policy_indices), hidden_size, device=self.device),
                    )

                while True:
                    policy_obs = torch.as_tensor(obs[policy_indices], device=self.device)
                    if self.use_rnn and lstm_state is not None:
                        logits, _ = policy.forward_eval(policy_obs, lstm_state)
                    else:
                        logits, _ = policy.forward_eval(policy_obs)

                    actions_tensor, _, _ = pufferlib.pytorch.sample_logits(logits)
                    policy_actions = actions_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)

                    joint_actions = np.zeros(self.num_agents, dtype=np.int64)
                    for idx, act in zip(policy_indices, policy_actions):
                        joint_actions[idx] = act

                    for idx in opponent_indices:
                        joint_actions[idx] = opponent.act(obs[idx])

                    obs, rewards, terminals, truncations, _ = env.step(joint_actions)
                    episode_returns += rewards

                    if self.use_rnn and lstm_state is not None:
                        done = torch.as_tensor(terminals[policy_indices], device=self.device, dtype=torch.bool)
                        for key in ("lstm_h", "lstm_c"):
                            lstm_state[key][:, :] = torch.where(
                                done.view(-1, 1),
                                torch.zeros_like(lstm_state[key]),
                                lstm_state[key],
                            )

                    if np.all(terminals):
                        policy_return = float(episode_returns[policy_indices].mean())
                        opponent_return = float(episode_returns[opponent_indices].mean())
                        result.return_total += policy_return

                        if policy_return > opponent_return + 1e-6:
                            result.wins += 1
                        elif opponent_return > policy_return + 1e-6:
                            result.losses += 1
                        else:
                            result.draws += 1
                        break
        finally:
            env.close()

        return result
