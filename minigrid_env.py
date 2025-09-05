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
        
        # Override action space to only include directional movement actions
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left

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
        self.goal_pos = (gx, gy)

        # agent spawn
        self.place_agent()

    def step(self, action):
        # First, rotate the agent to face the desired direction
        target_dir = action  # 0=up, 1=right, 2=down, 3=left
        
        # Calculate how many left turns needed to face target direction
        current_dir = self.agent_dir
        turns_needed = (target_dir - current_dir) % 4
        
        # Execute the turns
        for _ in range(turns_needed):
            super().step(self.actions.left)

        
        # Now move forward in the desired direction
        obs, reward, terminated, truncated, info = super().step(self.actions.forward)
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


def get_room_quadrant(pos, width, height):
    """Determine which quadrant/room a position is in"""
    mid_w, mid_h = width // 2, height // 2
    x, y = pos
    if x < mid_w and y < mid_h:
        return 0  # top-left
    elif x >= mid_w and y < mid_h:
        return 1  # top-right
    elif x < mid_w and y >= mid_h:
        return 2  # bottom-left
    else:
        return 3  # bottom-right

def get_room_center(quadrant, width, height):
        """Get approximate center position of a room quadrant"""
        mid_w, mid_h = width // 2, height // 2
        if quadrant == 0:  # top-left
            return (mid_w // 2, mid_h // 2)
        elif quadrant == 1:  # top-right
            return (mid_w + mid_w // 2, mid_h // 2)
        elif quadrant == 2:  # bottom-left
            return (mid_w // 2, mid_h + mid_h // 2)
        else:  # bottom-right
            return (mid_w + mid_w // 2, mid_h + mid_h // 2)
        

def run_explore_policy_four_rooms(env, max_T, step_and_record, act_random):
    """Systematic exploration policy for FourRoomsMemoryEnv"""
    visited_rooms = set()
    total_steps = 0
    
    def explore_room_systematically(env, max_steps_in_room=20):
        """Systematically explore current room with diverse actions"""
        actions_taken = 0
        visited_positions = set()
        
        # Define exploration pattern: try to cover the room with different movement patterns
        exploration_actions = [0, 1, 2, 3] * 5  # repeat each direction 5 times
        random.shuffle(exploration_actions)
        
        for action in exploration_actions:
            if actions_taken >= max_steps_in_room:
                break
                
            current_pos = tuple(env.agent_pos)
            step_and_record(action)
            actions_taken += 1
            
            # Track visited positions
            visited_positions.add(current_pos)
            
            # If we hit a wall or didn't move, try a different action
            if tuple(env.agent_pos) == current_pos:
                continue
                
        return actions_taken
    
    while total_steps < max_T:
        current_room = get_room_quadrant(env.agent_pos, env.width, env.height)
        
        # If we haven't explored this room much, do systematic exploration
        if current_room not in visited_rooms or len(visited_rooms) < 4:
            steps_taken = explore_room_systematically(env, max_steps_in_room=15)
            total_steps += steps_taken
            visited_rooms.add(current_room)
            
            step_and_record(0)
            
        else:
            # Try to navigate to an unexplored room or do random exploration
            unexplored_rooms = set([0, 1, 2, 3]) - visited_rooms
            if unexplored_rooms:
                target_room = random.choice(list(unexplored_rooms))
                target_pos = get_room_center(target_room, env.width, env.height)
                
                # Use BFS to navigate to target room
                path = bfs_shortest_path(env.grid, tuple(env.agent_pos), target_pos)
                if path and len(path) > 1:
                    planned_actions = plan_actions_from_path(env.agent_dir, path)
                    for action in planned_actions[:3]:  # Limit path following
                        if total_steps >= max_T:
                            break
                        step_and_record(action)
                        total_steps += 1
                else:
                    # Fallback to random action
                    step_and_record(act_random())
                    total_steps += 1
            else:
                # All rooms visited, do some random exploration
                step_and_record(act_random())
                total_steps += 1


def run_bfs_policy_four_rooms(env, max_T, step_and_record, act_random):
    """BFS-based optimal navigation policy for FourRoomsMemoryEnv"""
    total_steps = 0
    goal_pos = env.goal_pos
    goal_reached = False
    current_target = goal_pos
    
    while total_steps < max_T:
        current_pos = tuple(env.agent_pos)
        
        # Check if we've reached the current target
        if current_pos == current_target:
            if not goal_reached and current_target == goal_pos:
                goal_reached = True
            
            # Select a new random target point on the grid
            while True:
                # Try to find a valid random position
                rx = np.random.randint(1, env.width - 1)
                ry = np.random.randint(1, env.height - 1)
                obj = env.grid.get(rx, ry)
                if obj is None:  # Empty cell
                    current_target = (rx, ry)
                    break
        
        # Find shortest path to current target
        path = bfs_shortest_path(env.grid, current_pos, current_target)
        if path and len(path) > 1:
            # Convert path to actions and execute one step
            planned_actions = plan_actions_from_path(env.agent_dir, path)
            if planned_actions:
                # Execute the first action from the planned path
                action = planned_actions[0]
                step_and_record(action)
                total_steps += 1
            else:
                # No valid actions, try random
                step_and_record(act_random())
                total_steps += 1
        else:
            # No path found, try random action
            step_and_record(act_random())
            total_steps += 1


def run_bfs_policy_ten_rooms(env, max_T, step_and_record, act_random):
    """BFS-based optimal navigation policy for TenRoomsMemoryEnv"""
    total_steps = 0
    
    # Find the goal position (should be in the last room)
    goal_pos = None
    for x in range(env.width - 2, 0, -1):
        for y in range(1, env.height - 1):
            obj = env.grid.get(x, y)
            if obj is not None and hasattr(obj, 'type') and obj.type == 'goal':
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    
    if goal_pos:
        goal_reached = False
        current_target = goal_pos
        
        while total_steps < max_T:
            current_pos = tuple(env.agent_pos)
            
            # Check if we've reached the current target
            if current_pos == current_target:
                if not goal_reached and current_target == goal_pos:
                    goal_reached = True
                
                # Select a new random target point on the grid
                while True:
                    # Try to find a valid random position
                    rx = np.random.randint(1, env.width - 1)
                    ry = np.random.randint(1, env.height - 1)
                    obj = env.grid.get(rx, ry)
                    if obj is None:  # Empty cell
                        current_target = (rx, ry)
                        break
            
            # Find shortest path to current target
            path = bfs_shortest_path(env.grid, current_pos, current_target)
            if path and len(path) > 1:
                # Convert path to actions and execute one step
                planned_actions = plan_actions_from_path(env.agent_dir, path)
                if planned_actions:
                    # Execute the first action from the planned path
                    action = planned_actions[0]
                    step_and_record(action)
                    total_steps += 1
                else:
                    # No valid actions, try random
                    step_and_record(act_random())
                    total_steps += 1
            else:
                # No path found, try random action
                step_and_record(act_random())
                total_steps += 1
    else:
        # Goal not found, fall back to random
        for t in range(max_T):
            step_and_record(act_random())


def run_bfs_policy_multi_doors_keys(env, max_T, step_and_record, act_random):
    """BFS-based optimal navigation policy for MultiDoorsKeysEnv"""
    total_steps = 0
    key_picked_up = False
    target_door_pos = None
    door_unlocked = False
    current_target = None
    
    # Get target door position
    if hasattr(env, 'door_positions') and env.goal_color_idx is not None:
        target_door_pos, _ = env.door_positions[env.goal_color_idx]
    
    while total_steps < max_T:
        current_pos = tuple(env.agent_pos)
        
        # Check if we've reached the current target
        if current_target and current_pos == current_target:
            # Select a new random target point on the grid
            while True:
                # Try to find a valid random position
                rx = np.random.randint(1, env.width - 1)
                ry = np.random.randint(1, env.height - 1)
                obj = env.grid.get(rx, ry)
                if obj is None:  # Empty cell
                    current_target = (rx, ry)
                    break
        
        if not key_picked_up:
            # Phase 1: Find and pick up the correct key
            # Find the key that matches the target door color
            target_key_pos = None
            for x in range(1, env.width // 2):  # Keys are on left half
                for y in range(1, env.height - 1):
                    obj = env.grid.get(x, y)
                    if (obj is not None and hasattr(obj, 'type') and 
                        obj.type == 'key' and hasattr(obj, 'color') and 
                        obj.color == env.goal_color):
                        target_key_pos = (x, y)
                        break
                if target_key_pos:
                    break
            
            if target_key_pos:
                current_target = target_key_pos
                # Navigate to the key
                path = bfs_shortest_path(env.grid, current_pos, target_key_pos)
                if path and len(path) > 1:
                    planned_actions = plan_actions_from_path(env.agent_dir, path)
                    if planned_actions:
                        action = planned_actions[0]
                        step_and_record(action)
                        total_steps += 1
                        
                        # Check if we're now carrying the key
                        if env.carrying is not None and hasattr(env.carrying, 'color') and env.carrying.color == env.goal_color:
                            key_picked_up = True
                    else:
                        step_and_record(act_random())
                        total_steps += 1
                else:
                    step_and_record(act_random())
                    total_steps += 1
            else:
                # Key not found, try random
                step_and_record(act_random())
                total_steps += 1
        elif not door_unlocked:
            # Phase 2: Navigate to target door and unlock it
            if target_door_pos:
                current_target = target_door_pos
                path = bfs_shortest_path(env.grid, current_pos, target_door_pos)
                if path and len(path) > 1:
                    planned_actions = plan_actions_from_path(env.agent_dir, path)
                    if planned_actions:
                        action = planned_actions[0]
                        step_and_record(action)
                        total_steps += 1
                        
                        # Check if we're in front of the target door and can unlock it
                        if (tuple(env.agent_pos) == target_door_pos and 
                            env.carrying is not None and hasattr(env.carrying, 'color') and 
                            env.carrying.color == env.goal_color):
                            # Try to unlock the door
                            step_and_record(5)  # toggle action
                            total_steps += 1
                            door_unlocked = True
                    else:
                        step_and_record(act_random())
                        total_steps += 1
                else:
                    step_and_record(act_random())
                    total_steps += 1
            else:
                step_and_record(act_random())
                total_steps += 1
        else:
            # Phase 3: Continue to random points after completing the task
            if not current_target:
                # Select a new random target point on the grid
                while True:
                    # Try to find a valid random position
                    rx = np.random.randint(1, env.width - 1)
                    ry = np.random.randint(1, env.height - 1)
                    obj = env.grid.get(rx, ry)
                    if obj is None:  # Empty cell
                        current_target = (rx, ry)
                        break
            
            # Navigate to current target
            path = bfs_shortest_path(env.grid, current_pos, current_target)
            if path and len(path) > 1:
                planned_actions = plan_actions_from_path(env.agent_dir, path)
                if planned_actions:
                    action = planned_actions[0]
                    step_and_record(action)
                    total_steps += 1
                else:
                    step_and_record(act_random())
                    total_steps += 1
            else:
                step_and_record(act_random())
                total_steps += 1


def run_random_policy(env, max_T, step_and_record, act_random):
    """Random policy (default fallback)"""
    for t in range(max_T):
        step_and_record(act_random())


def run_episode(
    env: MiniGridEnv,
    max_steps: Optional[int] = None,
    goal_img: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    policy: str = "random",
) -> Trajectory:
    """
    Run one episode, return a Trajectory. 
    Policy options:
    - 'random': Random actions
    - 'explore': Systematic exploration for FourRoomsMemoryEnv (covers all rooms and actions)
    - 'bfs': BFS-based navigation (for all envs)
    """
    obs_list, act_list, rew_list, done_list, info_list = [], [], [], [], []
    rng_seed = seed

    obs, _ = env.reset(seed=seed)
    obs_list.append(obs["image"])
    max_T = max_steps or env.max_steps

    def step_and_record(action):
        """Helper to step environment and record trajectory data"""
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs["image"])
        act_list.append(action)

    def act_random():
        """Random action fallback"""
        return env.action_space.sample()

    # Main episode loop - delegate to policy-specific functions
    if policy == "explore" and isinstance(env, FourRoomsMemoryEnv):
        run_explore_policy_four_rooms(env, max_T, step_and_record, act_random)
    elif policy == "bfs":
        if isinstance(env, FourRoomsMemoryEnv):
            run_bfs_policy_four_rooms(env, max_T, step_and_record, act_random)
        elif isinstance(env, TenRoomsMemoryEnv):
            run_bfs_policy_ten_rooms(env, max_T, step_and_record, act_random)
        elif isinstance(env, MultiDoorsKeysEnv):
            run_bfs_policy_multi_doors_keys(env, max_T, step_and_record, act_random)
        else:
            # Unknown environment, fall back to random
            run_random_policy(env, max_T, step_and_record, act_random)
    else:
        # Random policy (default)
        run_random_policy(env, max_T, step_and_record, act_random)
    
    # cap the length of the trajectory to max_T
    obs_list = obs_list[:max_T]
    act_list = act_list[:max_T]

    obs_list = np.stack(obs_list, axis=0) # [max_T, H, W, 3]
    act_list = np.array(act_list)

        
    traj = Trajectory(
        observations=obs_list,
        actions=act_list,
    )
    return traj


# -------------------------
# Dataset writing
# -------------------------
def save_trajectories_npy(trajectories: List[Trajectory], out_dir: str, chunk_idx: int):
    """Save a single chunk of trajectories to separate NPY files for efficient memmap loading."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Stack trajectories into single arrays
    observations = np.stack([t.observations for t in trajectories])  # (N, T, H, W, 3)
    actions = np.stack([t.actions for t in trajectories])           # (N, T)
    
    # Save as separate NPY files
    obs_path = os.path.join(out_dir, f"observations_{chunk_idx:04d}.npy")
    act_path = os.path.join(out_dir, f"actions_{chunk_idx:04d}.npy")
    
    np.save(obs_path, observations)
    np.save(act_path, actions)
    
    return obs_path, act_path





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
    env_ctor, n_episodes: int = 100, max_steps: int = 100, seed: Optional[int] = None, policy: str = "random"
) -> EvalStats:
    successes, returns, lengths = [], [], []
    for _ in tqdm(range(n_episodes), desc=f"Eval {env_ctor.__name__}"):
        env = env_ctor()
        goal_img = None
        if isinstance(env, MultiDoorsKeysEnv):
            # capture the goal context image once per episode
            goal_img = env.get_goal_image()
        traj = run_episode(env, max_steps=max_steps, goal_img=goal_img, seed=seed, policy=policy)
        returns.append(traj.rewards.sum().item())
        lengths.append(len(traj.actions))
        # define success:
        if isinstance(env, MultiDoorsKeysEnv):
            # success if target door is open at the end
            ((dx, dy), _) = env.door_positions[env.goal_color_idx]
            door = env.grid.get(dx, dy)
            successes.append(int(isinstance(door, Door) and door.is_open) and lengths[-1] <= max_steps)
        else:
            # success if got reward >= 1
            successes.append(int(traj.rewards.sum() >= 1.0) and lengths[-1] <= max_steps)
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
    g.add_argument("--output-dir", type=str, default="minigrid_env")
    g.add_argument("--policy", choices=["random", "bfs", "explore"], default="random")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-steps", type=int, default=100)
    g.add_argument("--episodes-per-chunk", type=int, default=100)
    # evaluate
    e = sub.add_parser("eval", help="Evaluate scripted policies")
    e.add_argument(
        "--env", choices=["four_rooms", "ten_rooms", "mdk"], required=True
    )
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--policy", choices=["random", "bfs", "explore"], default="random")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.env == "four_rooms":
        ctor = lambda: make_four_rooms()
    elif args.env == "ten_rooms":
        ctor = lambda: make_ten_rooms()
    else:  # mdk
        ctor = lambda: make_multi_doors_keys()

    dataset_dir = os.environ["DATASET_DIR"]
    assert dataset_dir is not None, "DATASET_DIR must be set"
    output_path = os.path.join(dataset_dir, args.output_dir, f"{args.env}_{args.policy}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize chunking variables
    current_chunk = []
    chunk_idx = 0
    total_episodes = 0
    
    for episode_idx in tqdm(range(args.episodes), desc="Generating"):
        env = ctor()
        traj = run_episode(env, max_steps=args.max_steps, seed=args.seed, policy=args.policy)
        current_chunk.append(traj)
        env.close()
        total_episodes += 1
        
        # Save chunk when it reaches the target size
        if len(current_chunk) >= args.episodes_per_chunk:
            obs_path, act_path = save_trajectories_npy(current_chunk, output_path, chunk_idx)
      
            print(f"Saved chunk {chunk_idx} with {len(current_chunk)} episodes")
            current_chunk = []
            chunk_idx += 1
    
    # Save final partial chunk if it has any episodes
    if current_chunk:
        obs_path, act_path = save_trajectories_npy(current_chunk, output_path, chunk_idx)

        print(f"Saved final chunk {chunk_idx} with {len(current_chunk)} episodes")
        chunk_idx += 1
    
    # Create index file
    index = {
        'total_episodes': total_episodes,
        'episodes_per_chunk': args.episodes_per_chunk,
        'n_chunks': chunk_idx,
        'seed': args.seed,
        'policy': args.policy,
        'max_steps': args.max_steps,
        'episodes_per_chunk': args.episodes_per_chunk,
        'episodes': args.episodes,
        'output_dir': args.output_dir,
        'env': args.env,
    }
    
    index_path = os.path.join(output_path, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Saved {total_episodes} episodes in {chunk_idx} chunks to {output_path}")


if __name__ == "__main__":
    main()
