import collections
import copy
import os
import select
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import pufferlib
from pufferlib.ocean.tron import binding
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
        total_games = max(self.wins + self.draws + self.losses, 1)
        return {
            "win_rate": self.wins / total_games,
            "draw_rate": self.draws / total_games,
            "loss_rate": self.losses / total_games,
            "avg_return": self.return_total / total_games,
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

    def begin_episode(self, env: Tron, opponent_indices: List[int], policy_indices: List[int]) -> None:
        """Optional hook invoked at the start of each evaluation episode."""

    def end_episode(self) -> None:
        """Optional hook invoked when an evaluation episode terminates."""

    def act_with_env(self, env: Tron, observation: np.ndarray, agent_index: int) -> int:
        """Override to access the full environment state when selecting an action."""
        return self.act(observation)

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


class ClassicTronBotHeuristic(TronHeuristic):
    """Wrapper around the classic a1k0n Tron bot binary."""

    MOVE_TO_DIR = {1: 0, 2: 1, 3: 2, 4: 3}

    def __init__(self, vision: int, bot_path: str, env_index: int = 0, per_move_timeout: float = 0.1):
        super().__init__(vision)
        self.bot_path = os.path.abspath(bot_path)
        self.env_index = env_index
        self.per_move_timeout = max(0.0, float(per_move_timeout))
        self.process: Optional[subprocess.Popen] = None
        self._opponent_indices: List[int] = []
        self._policy_indices: List[int] = []
        self._agents_per_env: int = 0
        self._timeout_misses = 0
        self._warning_emitted = False

    def __del__(self):
        self._terminate_process()

    def begin_episode(self, env: Tron, opponent_indices: List[int], policy_indices: List[int]) -> None:
        if not opponent_indices:
            raise ValueError("Classic Tron bot heuristic requires at least one controlled agent index.")

        self._opponent_indices = list(opponent_indices)
        self._policy_indices = list(policy_indices)
        self._agents_per_env = getattr(env, "_agents_per_env", env.num_agents)
        self._timeout_misses = 0
        self._warning_emitted = False
        self._start_process()

    def _start_process(self) -> None:
        self._terminate_process()
        if not os.path.isfile(self.bot_path):
            raise FileNotFoundError(f"Classic Tron bot not found at '{self.bot_path}'")

        cwd = os.path.dirname(self.bot_path) or None
        try:
            self.process = subprocess.Popen(
                [self.bot_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=cwd,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to launch Classic Tron bot from '{self.bot_path}': {exc}") from exc
        self._timeout_misses = 0

    def end_episode(self) -> None:
        self._terminate_process()

    def act_with_env(self, env: Tron, observation: np.ndarray, agent_index: int) -> int:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Classic Tron bot process is not running. Did begin_episode fail?")

        board_payload = self._build_board_payload(env)
        try:
            self.process.stdin.write(board_payload)
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            if not self._warning_emitted:
                print("Warning: Classic Tron bot process closed; restarting and defaulting to forward action.")
                self._warning_emitted = True
            try:
                self._start_process()
            except Exception:
                return ACTION_FORWARD
            return ACTION_FORWARD

        move_line = self._readline_with_timeout(self.per_move_timeout)
        if not move_line:
            self._timeout_misses += 1
            if not self._warning_emitted:
                print("Warning: Classic Tron bot timed out; defaulting to forward action.")
                self._warning_emitted = True
            return ACTION_FORWARD

        move_line = move_line.strip()
        try:
            tron_move = int(move_line)
        except ValueError as exc:
            self._timeout_misses += 1
            if not self._warning_emitted:
                print(f"Warning: Classic Tron bot returned '{move_line}' – defaulting to forward action.")
                self._warning_emitted = True
            return ACTION_FORWARD

        return self._convert_move_to_action(tron_move, observation)

    def _convert_move_to_action(self, tron_move: int, observation: np.ndarray) -> int:
        current_dir = _parse_orientation(observation[-5:])
        target_dir = self.MOVE_TO_DIR.get(tron_move, current_dir)

        if target_dir == current_dir:
            return ACTION_FORWARD
        if target_dir == ((current_dir + 1) & 3):
            return ACTION_RIGHT
        if target_dir == ((current_dir + 3) & 3):
            return ACTION_LEFT

        # U-turns or unexpected outputs: default to turning right to avoid stalling.
        return ACTION_RIGHT

    def _build_board_payload(self, env: Tron) -> str:
        if not self._opponent_indices:
            raise RuntimeError("Classic Tron bot heuristic requires at least one opponent agent index")

        state = binding.env_get(env._c_env_handles[self.env_index])
        width = int(state["width"])
        height = int(state["height"])

        trail_owner = np.frombuffer(state["trail_owner"], dtype=np.uint8).reshape(height, width)
        head_owner = np.frombuffer(state["head_owner"], dtype=np.uint8).reshape(height, width)
        alive = np.frombuffer(state["alive"], dtype=np.uint8)

        grid = np.full((height, width), " ", dtype="<U1")
        grid[trail_owner > 0] = "#"
        grid[0, :] = "#"
        grid[-1, :] = "#"
        grid[:, 0] = "#"
        grid[:, -1] = "#"

        bot_agent_id = self._map_agent_index_to_local(self._opponent_indices[0])
        bot_position = self._find_head_position(head_owner, bot_agent_id)
        if bot_position and alive[bot_agent_id - 1]:
            y, x = bot_position
            grid[y, x] = "1"

        for idx in self._policy_indices:
            opp_agent_id = self._map_agent_index_to_local(idx)
            if opp_agent_id == bot_agent_id:
                continue
            pos = self._find_head_position(head_owner, opp_agent_id)
            if pos and alive[opp_agent_id - 1]:
                y, x = pos
                grid[y, x] = "2"

        lines = ["".join(row.tolist()) for row in grid]
        return f"{width} {height}\n" + "\n".join(lines) + "\n"

    def _map_agent_index_to_local(self, agent_index: int) -> int:
        if self._agents_per_env <= 0:
            return (agent_index % 2) + 1
        return (agent_index % self._agents_per_env) + 1

    @staticmethod
    def _find_head_position(head_owner: np.ndarray, agent_id: int) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(head_owner == agent_id)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0])

    def _readline_with_timeout(self, timeout: float) -> Optional[str]:
        if timeout <= 0.0:
            return self.process.stdout.readline()

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            read_ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if read_ready:
                return self.process.stdout.readline()
            if self.process.poll() is not None:
                break
        return None

    def _terminate_process(self) -> None:
        if self.process is None:
            return

        try:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            self.process.terminate()
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.process = None


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
        map_width_override = config.get("heuristic_eval_map_width")
        map_height_override = config.get("heuristic_eval_map_height")
        max_round_steps_override = config.get("heuristic_eval_max_round_steps")
        if map_width_override is not None:
            self.env_args["map_width"] = int(map_width_override)
        if map_height_override is not None:
            self.env_args["map_height"] = int(map_height_override)
        if max_round_steps_override is not None:
            self.env_args["max_round_steps"] = int(max_round_steps_override)

        vision = env_args.get("vision_size", 9)
        self.heuristics: Dict[str, TronHeuristic] = {}
        tronbot_path = (
            config.get("heuristic_eval_tronbot_path")
            or env_args.get("heuristic_eval_tronbot_path")
            or config.get("classic_tronbot_path")
        )
        if not tronbot_path:
            builtin_candidate = os.path.join(
                os.path.dirname(__file__), "../../../resources/tronbot/MyTronBot"
            )
            builtin_candidate = os.path.abspath(builtin_candidate)
            if os.path.isfile(builtin_candidate) and os.access(builtin_candidate, os.X_OK):
                tronbot_path = builtin_candidate
        if not tronbot_path:
            external_candidate = os.path.join(
                os.path.dirname(__file__), "../../../tronbot/cpp/MyTronBot"
            )
            external_candidate = os.path.abspath(external_candidate)
            if os.path.isfile(external_candidate) and os.access(external_candidate, os.X_OK):
                tronbot_path = external_candidate
        if tronbot_path:
            timeout_cfg = (
                config.get("heuristic_eval_tronbot_timeout")
                or env_args.get("heuristic_eval_tronbot_timeout")
            )
            per_move_timeout = float(timeout_cfg) if timeout_cfg not in (None, "") else 0.1
            self.heuristics["classic"] = ClassicTronBotHeuristic(
                vision,
                tronbot_path,
                per_move_timeout=per_move_timeout,
            )
            if self.num_agents > 2:
                self.num_agents = 2
                self.env_args["num_agents"] = 2

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

    def generate_video(
        self,
        policy: torch.nn.Module,
        opponent_name: str,
        max_frames: int,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        opponent = self.heuristics.get(opponent_name)
        if opponent is None:
            raise KeyError(f"No heuristic opponent named '{opponent_name}'")

        env = Tron(**self.env_args)
        frames: List[np.ndarray] = []

        policy_model = getattr(policy, "module", policy)
        training_state = policy_model.training
        policy.eval()
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        policy_indices = [0]
        opponent_indices = [i for i in range(self.num_agents) if i not in policy_indices]
        hidden_size = getattr(policy_model, "hidden_size", None)

        try:
            opponent.reset()
            rollout_seed = self.base_seed if seed is None else seed
            obs, _ = env.reset(seed=rollout_seed)

            lstm_state = None
            if self.use_rnn and hidden_size is not None:
                lstm_state = dict(
                    lstm_h=torch.zeros(len(policy_indices), hidden_size, device=self.device),
                    lstm_c=torch.zeros(len(policy_indices), hidden_size, device=self.device),
                )

            opponent.begin_episode(env, opponent_indices, policy_indices)

            steps = 0
            while steps < max_frames:
                frame = env.rgb_array(env_index=0)
                frames.append(np.array(frame, copy=True))

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
                    joint_actions[idx] = opponent.act_with_env(env, obs[idx], idx)

                obs, _, terminals, _, _ = env.step(joint_actions)
                steps += 1

                if self.use_rnn and lstm_state is not None:
                    done = torch.as_tensor(terminals[policy_indices], device=self.device, dtype=torch.bool)
                    for key in ("lstm_h", "lstm_c"):
                        lstm_state[key][:, :] = torch.where(
                            done.view(-1, 1),
                            torch.zeros_like(lstm_state[key]),
                            lstm_state[key],
                        )

                if np.all(terminals):
                    frame = env.rgb_array(env_index=0)
                    frames.append(np.array(frame, copy=True))
                    break
        finally:
            try:
                opponent.end_episode()
            except Exception:
                pass
            env.close()
            torch.set_grad_enabled(prev_grad)
            if training_state:
                policy.train()

        return frames

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

                opponent.begin_episode(env, opponent_indices, policy_indices)
                try:
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
                            joint_actions[idx] = opponent.act_with_env(env, obs[idx], idx)

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
                    opponent.end_episode()
        finally:
            env.close()

        return result
