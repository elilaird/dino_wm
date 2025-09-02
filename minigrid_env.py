import os
import math
import json
import uuid
import time
import random
import imageio
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Door, Key, Goal, Ball
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace


# -------------------------
# Utility: BFS over free grid
# -------------------------
def bfs_shortest_path(
    grid, start: Tuple[int, int], goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    BFS on MiniGrid grid coordinates. Returns list of (x,y) from start -> goal (inclusive), or None.
    Treats closed doors and walls as obstacles. Open doors are traversable.
    """
    width, height = grid.width, grid.height
    sx, sy = start
    gx, gy = goal
    q = [(sx, sy)]
    parent = {(sx, sy): None}

    def neighbors(x, y):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            obj = grid.get(nx, ny)
            if obj is None:
                yield (nx, ny)
            else:
                # traversable if not a wall and not a closed door
                if isinstance(obj, Wall):
                    continue
                if isinstance(obj, Door) and not obj.is_open:
                    continue
                yield (nx, ny)

    while q:
        x, y = q.pop(0)
        if (x, y) == (gx, gy):
            # reconstruct
            path = [(x, y)]
            while parent[(x, y)] is not None:
                x, y = parent[(x, y)]
                path.append((x, y))
            return list(reversed(path))
        for nx, ny in neighbors(x, y):
            if (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None


def direction_to_face(curr: Tuple[int, int], nxt: Tuple[int, int]) -> int:
    """
    Convert a move (curr -> next) into a compass dir: 0=right, 1=down, 2=left, 3=up (MiniGrid convention).
    """
    (x, y), (nx, ny) = curr, nxt
    dx, dy = nx - x, ny - y
    if dx == 1 and dy == 0:  # east
        return 0
    if dx == 0 and dy == 1:  # south
        return 1
    if dx == -1 and dy == 0:  # west
        return 2
    if dx == 0 and dy == -1:  # north
        return 3
    return 0


def plan_actions_from_path(
    agent_dir: int, path: List[Tuple[int, int]]
) -> List[int]:
    """
    Turn a coordinate path into MiniGrid actions (left=0, right=1, forward=2, pickup=3, drop=4, toggle=5, done=6).
    We only use left/right/forward here; door interactions handled separately.
    """
    actions = []
    if not path or len(path) < 2:
        return actions
    cur_dir = agent_dir
    for i in range(len(path) - 1):
        face = direction_to_face(path[i], path[i + 1])
        # rotate to face the right direction
        diff = (face - cur_dir) % 4
        if diff == 1:
            actions.append(1)  # right
        elif diff == 2:
            actions.extend([1, 1])  # 180 turn
        elif diff == 3:
            actions.append(0)  # left
        cur_dir = face
        actions.append(2)  # forward
    return actions


# -------------------------
# Base: simple palette helpers
# -------------------------
DOOR_COLORS = ["red", "green", "blue", "yellow", "purple"]
KEY_COLORS = DOOR_COLORS


# -------------------------
# Four Rooms Memory Env
# -------------------------
class FourRoomsMemoryEnv(MiniGridEnv):
    """
    Classic Four Rooms layout. The 'memory' aspect comes from long navigation & partial observability (agent view).
    Success is reaching the Goal.
    """

    def __init__(
        self,
        world_size: int = 17,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: Optional[str] = "rgb_array",
        obs_mode: str = "top_down", # "top_down" or "pov"
        tile_size: int = 14, 
        seed: Optional[int] = None,
        
    ):
        assert world_size % 2 == 1 and world_size >= 7, f"Size must be an odd number >= 7, got {world_size}"
        self.obs_mode = obs_mode
        self.world_size = world_size
        mission_space = MissionSpace(
            mission_func=lambda: "Reach the green goal"
        )
        super().__init__(
            mission_space=mission_space,
            width=world_size,
            height=world_size,
            max_steps=max_steps or (world_size * world_size),
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            tile_size=tile_size,
        )
        self.obs_mode = obs_mode
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed=None):
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # outer walls
        self.grid.wall_rect(0, 0, width, height)

        # internal walls to make 4 rooms
        mid_w = width // 2
        mid_h = height // 2
        self.grid.horz_wall(1, mid_h, width - 2)
        self.grid.vert_wall(mid_w, 1, height - 2)

        # each wall tuple (x, y, start, end)
        walls = (
            ("horz", 1, mid_h, 1, mid_w - 1),
            ("vert", mid_w, 1, 1, mid_h - 1),
            ("horz", 1, mid_h, mid_w + 1, width - 2),
            ("vert", mid_w, 1, mid_h + 1, height - 2),
        )
        # add random door openings in each of the 4 internal walls
        for wall, x, y, start, end in walls:
            pos = np.random.randint(start, end)  
            if wall == "horz":
                self.grid.set(pos, y, None)
            else:
                self.grid.set(x, pos, None)

        # goal in a random quadrant
        quadrant = np.random.randint(0, 4)
        gx = np.random.randint(1, mid_w - 1) + (quadrant % 2) * mid_w 
        gy = np.random.randint(1, mid_h - 1) + (quadrant // 2) * mid_h
        self.put_obj(Goal(), gx, gy)

        # agent spawn
        self.place_agent()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 1.0
            terminated = True
        return obs, reward, terminated, truncated, info

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        if self.obs_mode == "pov":
            image = self.get_pov_render(tile_size=self.tile_size)
        elif self.obs_mode == "top_down":
            image = self.get_full_render(highlight=False, tile_size=self.tile_size)
        else:
            image = image # encoded grid in MiniGrid format

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission,
        }

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed if seed is None else seed)
        return self.gen_obs(), {}


# -------------------------
# Ten Rooms (corridor chain) Env
# -------------------------
class TenRoomsMemoryEnv(MiniGridEnv):
    """
    10 small rooms in a chain (long navigation horizon). Reaching Goal tests long-term memory of path decisions.
    """

    def __init__(
        self,
        room_w: int = 7,
        n_rooms: int = 10,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: Optional[str] = "rgb_array",
        obs_mode: str = "top_down", # "top_down" or "pov"
        tile_size: int = 14,
    ):
        width = n_rooms * (room_w - 1) + 1
        height = room_w
        mission_space = MissionSpace(
            mission_func=lambda: "Find the goal at the far end"
        )
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps or (width * height),
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            tile_size=tile_size,
        )
        self.room_w = room_w
        self.n_rooms = n_rooms
        self.obs_mode = obs_mode
        self.tile_size = tile_size
        self._seed = seed
        self.set_seed(seed)

    def set_seed(self, seed=None):
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # carve 10 rooms with a 1-tile doorway in each separating wall
        x = 1
        for i in range(self.n_rooms):
            # right wall for room i (except last is outer wall)
            if i < self.n_rooms - 1:
                rx = x + self.room_w - 1
                # draw a wall
                for y in range(1, self.height - 1):
                    self.grid.set(rx, y, Wall())
                # opening at random y
                oy = np.random.randint(1, self.height - 2)
                self.grid.set(rx, oy, None)
            x += self.room_w - 1

        # goal in the last room at random spot
        gx = width - 2
        gy = np.random.randint(1, height - 2)
        self.put_obj(Goal(), gx, gy)

        # agent spawn in the first room
        self.place_agent(top=(1, 1), size=(self.room_w - 1, height - 2))

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        if self.obs_mode == "pov":
            image = self.get_pov_render(tile_size=self.tile_size)
        elif self.obs_mode == "top_down":
            image = self.get_full_render(highlight=False, tile_size=self.tile_size)
        else:
            image = image # encoded grid in MiniGrid format

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission,
        }

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed if seed is None else seed)
        return self.gen_obs(), {}


# -------------------------
# Multi Doors & Keys (context-dependent recall with goal image)
# -------------------------
class MultiDoorsKeysEnv(MiniGridEnv):
    """
    Place K keys (colored) and M doors (colored & locked). Goal is to unlock the door whose *goal image*
    (an unlocked door of that color) is shown to the agent as context (you’ll store this in the dataset).
    The world requires remembering which door to go to after picking the matching key.
    """

    def __init__(
        self,
        world_size: int = 13,
        n_keys: int = 3,
        n_doors: int = 3,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: Optional[str] = "rgb_array",
        obs_mode: str = "top_down", # "top_down" or "pov"
        tile_size: int = 14,
    ):
        mission_space = MissionSpace(
            mission_func=lambda: "Unlock the correct goal door using the matching key"
        )
        super().__init__(
            mission_space=mission_space,
            width=world_size,
            height=world_size,
            max_steps=max_steps or (world_size * world_size),
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            tile_size=tile_size,
        )
        self._world_size = world_size
        self.n_keys = min(n_keys, len(KEY_COLORS))
        self.n_doors = min(n_doors, len(DOOR_COLORS))
        self.colors = DOOR_COLORS[: max(self.n_keys, self.n_doors)]
        self.goal_color_idx = None  # which door color is the designated goal
        self.obs_mode = obs_mode
        self.tile_size = tile_size
        self._seed = seed
        self.set_seed(seed)

    def set_seed(self, seed=None):
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Scatter doors on the right wall, keys on the left half
        door_cols = np.random.choice(self.colors, size=self.n_doors, replace=False)
        key_cols = np.random.choice(self.colors, size=self.n_keys, replace=False)

        # Place doors on right wall, locked
        y_positions = np.random.choice(range(2, height - 2), size=self.n_doors, replace=False)
        self.door_positions = []
        for i, col in enumerate(door_cols):
            dy = y_positions[i]
            d = Door(col, is_locked=True)
            self.put_obj(d, width - 2, dy)
            self.door_positions.append(((width - 2, dy), col))

        # Place keys randomly on left half
        for col in key_cols:
            self.place_obj(
                Key(col),
                top=(1, 1),
                size=(width // 2, height - 2),
                max_tries=100,
            )

        # Agent
        self.place_agent()

        # Choose a goal door color from the placed doors
        self.goal_color_idx = np.random.randint(0, len(door_cols))
        self.goal_color = door_cols[self.goal_color_idx]

    def get_goal_image(self) -> np.ndarray:
        """
        Render a *goal image* showing the target door as unlocked (context image). Used for dataset/eval.
        We temporarily set that door to unlocked, render, then revert.
        """
        ((dx, dy), color) = self.door_positions[self.goal_color_idx]
        door: Door = self.grid.get(dx, dy)
        prev_locked = door.is_locked
        prev_open = door.is_open
        # emulate unlocked & open in the goal image to make it visually clear
        door.is_locked = False
        door.is_open = True
        img = self.render(mode="rgb_array")
        # revert
        door.is_locked = prev_locked
        door.is_open = prev_open
        return img

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # if facing a locked door but holding matching key, toggle opens it
        if action == self.actions.toggle:
            fwd_pos = self.front_pos
            fwd_obj = self.grid.get(*fwd_pos)
            if isinstance(fwd_obj, Door):
                if (
                    fwd_obj.is_locked
                    and self.carrying
                    and isinstance(self.carrying, Key)
                ):
                    if self.carrying.color == fwd_obj.color:
                        fwd_obj.is_locked = False
                        reward += 0.05  # shaping

        # success = the target door is open
        ((dx, dy), color) = self.door_positions[self.goal_color_idx]
        target: Door = self.grid.get(dx, dy)
        if isinstance(target, Door) and target.is_open:
            reward += 1.0
            terminated = True

        return obs, reward, terminated, truncated, info

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        if self.obs_mode == "pov":
            image = self.get_pov_render(tile_size=self.tile_size)
        elif self.obs_mode == "top_down":
            image = self.get_full_render(highlight=False, tile_size=self.tile_size)
        else:
            image = image # encoded grid in MiniGrid format

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission,
        }

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed if seed is None else seed)
        return self.gen_obs(), {}


# -------------------------
# Rollout helpers
# -------------------------
@dataclass
class Trajectory:
    observations: np.ndarray  # [T, H, W, 3]
    actions: np.ndarray  # [T]
    rewards: np.ndarray  # [T]
    dones: np.ndarray  # [T]
    infos: List[Dict[str, Any]]
    goal_image: Optional[np.ndarray] = None
    env_name: str = ""
    seed: int = 0


def run_episode(
    env: MiniGridEnv,
    policy: str = "bfs",
    max_steps: Optional[int] = None,
    goal_img: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Trajectory:
    """
    Run one episode, return a Trajectory. 'bfs' will try naive waypointing to the visible goal/door/key,
    falling back to random moves. For MultiDoorsKeys we do a simple two-stage script:
      1) pick up any key
      2) if key color == goal door color => navigate to that door and toggle
    """
    obs_list, act_list, rew_list, done_list, info_list = [], [], [], [], []
    rng_seed = seed

    obs, _ = env.reset(seed=seed)
    obs_list.append(obs["image"])
    T = 0
    max_T = max_steps or env.max_steps

    def act_random():
        return env.action_space.sample()

    def step_and_record(a):
        nonlocal T, obs_list, act_list, rew_list, done_list, info_list
        ob, rw, terminated, truncated, info = env.step(a)
        obs_list.append(ob["image"])
        act_list.append(a)
        rew_list.append(rw)
        done_list.append(terminated or truncated)
        info_list.append(info)
        T += 1
        return terminated or truncated

    if policy == "random":
        while T < max_T:
            if step_and_record(act_random()):
                break
    else:
        # "bfs" policy
        while T < max_T:
            # MultiDoorsKeys: scripted goal
            if isinstance(env, MultiDoorsKeysEnv):
                # if not carrying a key, navigate to nearest key
                if env.carrying is None:
                    # find keys
                    keys = []
                    for x in range(env.grid.width):
                        for y in range(env.grid.height):
                            obj = env.grid.get(x, y)
                            if isinstance(obj, Key):
                                keys.append((x, y))
                    if keys:
                        # naive nearest key by manhattan
                        keys.sort(
                            key=lambda p: abs(p[0] - env.agent_pos[0])
                            + abs(p[1] - env.agent_pos[1])
                        )
                        path = bfs_shortest_path(
                            env.grid, tuple(env.agent_pos), keys[0]
                        )
                        planned = (
                            plan_actions_from_path(env.agent_dir, path)
                            if path
                            else [act_random()]
                        )
                        if not planned:
                            planned = [act_random()]
                        a = planned[0]
                        if step_and_record(a):
                            break
                        # pickup if on key
                        if tuple(env.agent_pos) == keys[0]:
                            if step_and_record(env.actions.pickup):
                                break
                        continue
                # have a key: if color matches goal, navigate to that door, then toggle
                target_pos, target_color = env.door_positions[
                    env.goal_color_idx
                ]
                if (
                    env.carrying
                    and isinstance(env.carrying, Key)
                    and env.carrying.color == target_color
                ):
                    path = bfs_shortest_path(
                        env.grid, tuple(env.agent_pos), target_pos
                    )
                    planned = (
                        plan_actions_from_path(env.agent_dir, path)
                        if path
                        else [act_random()]
                    )
                    a = planned[0] if planned else act_random()
                    if step_and_record(a):
                        break
                    # attempt to toggle if in front of door
                    fwd = env.front_pos
                    if tuple(fwd) == target_pos:
                        if step_and_record(env.actions.toggle):
                            break
                    continue

            # generic “head toward goal if visible in grid, else random”
            # find Goal object
            goal_pos = None
            for x in range(env.grid.width):
                for y in range(env.grid.height):
                    obj = env.grid.get(x, y)
                    if isinstance(obj, Goal):
                        goal_pos = (x, y)
                        break
                if goal_pos:
                    break

            a = act_random()
            if goal_pos is not None:
                path = bfs_shortest_path(
                    env.grid, tuple(env.agent_pos), goal_pos
                )
                if path and len(path) > 1:
                    planned = plan_actions_from_path(env.agent_dir, path)
                    a = planned[0] if planned else act_random()

            if step_and_record(a):
                break

    traj = Trajectory(
        observations=np.asarray(obs_list, dtype=np.uint8),
        actions=np.asarray(act_list, dtype=np.int64),
        rewards=np.asarray(rew_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.bool_),
        infos=info_list,
        goal_image=goal_img,
        env_name=(
            env.spec.id if env.spec is not None else env.__class__.__name__
        ),
        seed=rng_seed,
    )
    return traj


# -------------------------
# Dataset writing
# -------------------------
def save_trajectories_npz(trajectories: List[Trajectory], out_path: str):
    pack = {
        "observations": [t.observations for t in trajectories],
        "actions": [t.actions for t in trajectories],
        "rewards": [t.rewards for t in trajectories],
        "dones": [t.dones for t in trajectories],
        "infos": [json.dumps(t.infos) for t in trajectories],
        "env_names": [t.env_name for t in trajectories],
        "seeds": [t.seed for t in trajectories],
        "goal_images": [
            (
                t.goal_image
                if t.goal_image is not None
                else np.array([], dtype=np.uint8)
            )
            for t in trajectories
        ],
    }
    np.savez_compressed(out_path, **pack)


# -------------------------
# Evaluation
# -------------------------
@dataclass
class EvalStats:
    n_episodes: int
    success_rate: float
    avg_return: float
    avg_length: float
    horizon_75th: float  # 75th percentile of episode length
    notes: str = ""


def evaluate_env(
    env_ctor, n_episodes: int = 100, policy: str = "bfs", seed: Optional[int] = None
) -> EvalStats:
    successes, returns, lengths = [], [], []
    for _ in tqdm(range(n_episodes), desc=f"Eval {env_ctor.__name__}"):
        env = env_ctor()
        goal_img = None
        if isinstance(env, MultiDoorsKeysEnv):
            # capture the goal context image once per episode
            goal_img = env.get_goal_image()
        traj = run_episode(env, policy=policy, goal_img=goal_img, seed=seed)
        returns.append(traj.rewards.sum().item())
        lengths.append(len(traj.actions))
        # define success:
        if isinstance(env, MultiDoorsKeysEnv):
            # success if target door is open at the end
            ((dx, dy), _) = env.door_positions[env.goal_color_idx]
            door = env.grid.get(dx, dy)
            successes.append(int(isinstance(door, Door) and door.is_open))
        else:
            # success if got reward >= 1
            successes.append(int(traj.rewards.sum() >= 1.0))
        env.close()

    sr = float(np.mean(successes)) if successes else 0.0
    avg_ret = float(np.mean(returns)) if returns else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    h75 = float(np.percentile(lengths, 75)) if lengths else 0.0
    return EvalStats(
        n_episodes=n_episodes,
        success_rate=sr,
        avg_return=avg_ret,
        avg_length=avg_len,
        horizon_75th=h75,
        notes="Success = unlocked goal door (MultiDoorsKeys) or reached Goal (Rooms).",
    )


# -------------------------
# Environment factory wrappers
# -------------------------
def make_four_rooms(world_size=17, obs_mode="top_down", tile_size=14, agent_view_size=7):
    return FourRoomsMemoryEnv(
        world_size=world_size, 
        obs_mode=obs_mode, 
        tile_size=tile_size, 
        agent_view_size=agent_view_size
    )


def make_ten_rooms(room_w=7, n_rooms=10, obs_mode="top_down", tile_size=14, agent_view_size=7):
    return TenRoomsMemoryEnv(
        room_w=room_w, 
        n_rooms=n_rooms, 
        obs_mode=obs_mode, 
        tile_size=tile_size, 
        agent_view_size=agent_view_size
    )


def make_multi_doors_keys(world_size=13, n_keys=3, n_doors=3, obs_mode="top_down", tile_size=14, agent_view_size=7):
    return MultiDoorsKeysEnv(
        world_size=world_size, 
        n_keys=n_keys, 
        n_doors=n_doors, 
        obs_mode=obs_mode, 
        tile_size=tile_size, 
        agent_view_size=agent_view_size
    )


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MiniGrid Memory Suite (envs, rollouts, datasets, eval)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate dataset
    g = sub.add_parser("generate", help="Generate offline datasets")
    g.add_argument(
        "--env", choices=["four_rooms", "ten_rooms", "mdk"], required=True
    )
    g.add_argument("--episodes", type=int, default=1000)
    g.add_argument("--out", type=str, default="dataset_shard.npz")
    g.add_argument("--policy", choices=["random", "bfs"], default="bfs")
    g.add_argument("--seed", type=int, default=None)

    # evaluate
    e = sub.add_parser("eval", help="Evaluate scripted policies")
    e.add_argument(
        "--env", choices=["four_rooms", "ten_rooms", "mdk"], required=True
    )
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--policy", choices=["random", "bfs"], default="bfs")

    args = parser.parse_args()

    if args.cmd == "generate":
        if args.seed is not None:
            np.random.seed(args.seed)

        if args.env == "four_rooms":
            ctor = lambda: make_four_rooms()
        elif args.env == "ten_rooms":
            ctor = lambda: make_ten_rooms()
        else:  # mdk
            ctor = lambda: make_multi_doors_keys()

        trajectories: List[Trajectory] = []
        for _ in tqdm(range(args.episodes), desc="Generating"):
            env = ctor()
            goal_img = (
                env.get_goal_image()
                if isinstance(env, MultiDoorsKeysEnv)
                else None
            )
            traj = run_episode(env, policy=args.policy, goal_img=goal_img, seed=args.seed)
            trajectories.append(traj)
            env.close()

        save_trajectories_npz(trajectories, args.out)
        print(f"Saved {len(trajectories)} episodes to {args.out}")

    elif args.cmd == "eval":
        if args.env == "four_rooms":
            ctor = make_four_rooms
        elif args.env == "ten_rooms":
            ctor = make_ten_rooms
        else:
            ctor = make_multi_doors_keys

        stats = evaluate_env(
            ctor, n_episodes=args.episodes, policy=args.policy, seed=args.seed
        )
        print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
