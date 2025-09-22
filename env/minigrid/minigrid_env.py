import math
import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from tqdm import tqdm

from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Door, Key, Goal, Ball, Box
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace


def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        full_dct[key] = np.stack(value)
    return full_dct


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

class CustomDoor(Door):
    # custom door for blocking view only
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_open = False
        self.is_locked = False

    def see_behind(self):
        return False

    def can_overlap(self):
        return True

    def toggle(self, env, pos):
        return True


class OverlapBall(Ball):
    def __init__(self, color="blue"):
        super().__init__(color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True


class OverlapKey(Key):
    def __init__(self, color="blue"):
        super().__init__(color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True


class OverlapBox(Box):
    def __init__(self, color="blue"):
        super().__init__(color)

    def can_overlap(self):
        return True

    def see_behind(self):
        return True
# -------------------------
# Four Rooms Memory Env
# -------------------------
class FourRoomsMemoryEnv(MiniGridEnv):
    """
    Classic Four Rooms layout with memory testing capabilities. The 'memory' aspect comes from:
    1. Long navigation & partial observability (agent view)
    2. Object placement and recall tasks
    3. Spatial memory challenges
    
    Memory test modes:
    - 'navigation': Basic navigation to goal
    - 'object_recall': Remember object locations after exploration
    - 'color_memory': Remember object colors and locations
    - 'sequential_memory': Remember sequence of object placements
    """

    def __init__(
        self,
        world_size: int = 17,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        agent_view_size: int = None,
        render_mode: Optional[str] = "rgb_array",
        obs_mode: str = "top_down", # "top_down" or "pov"
        tile_size: int = 14, 
        seed: Optional[int] = None,
        memory_test_mode: str = "navigation",  # "navigation", "object_recall", "color_memory", "sequential_memory"
        n_memory_objects: int = 3,  # Number of objects to place for memory tests
        memory_object_types: List[str] = None,  # Types of objects to place
    ):
        assert world_size % 2 == 1 and world_size >= 7, f"Size must be an odd number >= 7, got {world_size}"
        self.obs_mode = obs_mode
        self.world_size = world_size
        self.memory_test_mode = memory_test_mode
        self.n_memory_objects = n_memory_objects
        self.memory_object_types = memory_object_types or ["ball", "box", "key"]
        self.set_seed(seed)
        self.init_state = None
        self.agent_view_size = agent_view_size or math.ceil(world_size / 2)
        if self.agent_view_size % 2 == 0:
            self.agent_view_size += 1
        self.num_rooms = 4

        # Memory test state
        self.memory_objects = []  # List of (object, position, color) tuples
        self.memory_questions = []  # Questions about object locations/colors
        self.memory_phase = "exploration"  # "exploration", "question", "navigation"
        self.current_question_idx = 0

        # Define mission based on memory test mode
        mission_funcs = {
            "navigation": lambda: "Reach the green goal",
            "object_recall": lambda: "Remember object locations and answer questions",
            "color_memory": lambda: "Remember object colors and locations",
            "sequential_memory": lambda: "Remember the sequence of object placements"
        }
        mission_space = MissionSpace(mission_func=mission_funcs.get(memory_test_mode, mission_funcs["navigation"]))
        super().__init__(
            mission_space=mission_space,
            width=world_size,
            height=world_size,
            max_steps=max_steps or (world_size * world_size),
            see_through_walls=see_through_walls,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            tile_size=tile_size,
        )

        # Override action space to only include directional movement actions
        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: forward
        # possible proprio directions: 0: right, 1: down, 2: left, 3: up

    def set_seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed) 

    def sample_random_pos(self):
        mid_w = self.width // 2
        mid_h = self.height // 2
        quadrant = np.random.randint(0, 4)
        gx = np.random.randint(1, mid_w - 1) + (quadrant % 2) * mid_w 
        gy = np.random.randint(1, mid_h - 1) + (quadrant // 2) * mid_h
        return (np.int64(gx), np.int64(gy))

    def place_agent(self, init_state=None):
        if init_state is None:
            super().place_agent()
        else:
            self.grid.set(int(init_state[0]), int(init_state[1]), None)
            self.agent_pos = (np.int64(init_state[0]), np.int64(init_state[1]))
            self.agent_dir = self._get_inward_direction(self.agent_pos)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = {}
        info['state'] = self.agent_pos
        info['memory_phase'] = self.memory_phase

        # Handle memory testing phases
        if self.memory_test_mode != "navigation":
            info.update(self._handle_memory_phase(action))

        return obs, reward, terminated, truncated, info

    def step_multiple(self, actions):
        obses = []
        infos = []
        for action in actions:
            obs, _, _, _, info = self.step(action)
            obses.append(obs)
            infos.append(info)
        obses = aggregate_dct(obses)
        infos = aggregate_dct(infos)
        return obses, infos

    def rollout(self, seed, init_state, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        obs, state = self.prepare(seed, init_state)
        obses, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states

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
        obs = {
            "visual": image,
            "proprio": self._get_proprio(),
        }

        return obs

    def set_init_state(self, init_state):
        self.init_state = init_state

    def prepare(self, seed, init_state):
        self.set_init_state(init_state)
        obs, state = self.reset(seed) # calls _gen_grid under the hood
        return obs, state

    def reset(self, seed=None, options=None):
        self.set_seed(seed)
        obs, _ = super().reset(seed=self.seed if seed is None else seed)
        return obs, self.agent_pos

    def sample_random_init_goal_states(self, seed):
        # self.set_seed(seed)
        ax, ay = self.sample_random_pos()

        gx, gy = self.sample_random_pos()
        return (ax, ay), (gx, gy)

    def update_env(self, env_info):
        pass 

    def eval_state(self, goal_state, cur_state):
        gx, gy = int(goal_state[0]), int(goal_state[1])
        cx, cy = int(cur_state[0]), int(cur_state[1])
        success = (gx == cx) and (gy == cy)
        state_dist = abs(gx - cx) + abs(gy - cy)

        result = {
            'success': success,
            'state_dist': state_dist,
        }

        # Add memory-specific evaluation
        if self.memory_test_mode != "navigation":
            result.update(self._eval_memory_performance())

        return result

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

        # Place memory objects based on test mode
        if self.memory_test_mode != "navigation":
            self._place_memory_objects()

        # Place goal for navigation tasks
        if self.memory_test_mode == "navigation":
            gx, gy = self.sample_random_pos()
            self.put_obj(Goal(), gx, gy)
            self.goal_pos = (gx, gy)

        # agent spawn
        self.place_agent(init_state=self.init_state) 

    def _get_proprio(self):
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        dir = int(self.agent_dir)
        return np.array([x, y, dir])

    def _place_memory_objects(self):
        """Place memory objects in different rooms for testing"""
        self.memory_objects = []
        colors = ["red", "green", "blue", "yellow", "purple"]
        used_positions = set()  # Track used positions to avoid overlaps

        # Ensure objects are placed in different rooms
        room_positions = []
        for room in range(self.num_rooms):
            room_positions.append(self._get_room_positions(room))

        # Place objects in different rooms
        for i in range(self.n_memory_objects):
            room_idx = i % self.num_rooms
            available_positions = room_positions[room_idx]
            
            # Filter out already used positions
            available_positions = [pos for pos in available_positions if pos not in used_positions]
            
            if available_positions:
                pos = available_positions[
                    np.random.randint(0, len(available_positions))
                ]
                used_positions.add(pos)  # Mark position as used
                
                obj_type = self.memory_object_types[
                    i % len(self.memory_object_types)
                ]
                color = colors[i % len(colors)]

                # Create object based on type
                if obj_type == "ball":
                    obj = OverlapBall(color)
                elif obj_type == "box":
                    obj = OverlapBox(color)
                elif obj_type == "key":
                    obj = OverlapKey(color)
                else:
                    obj = OverlapBall(color)  # Default

                self.put_obj(obj, pos[0], pos[1])
                self.memory_objects.append((obj, pos, color, obj_type))

        # Generate memory questions
        self._generate_memory_questions()

    def _get_room_positions(self, room_idx):
        """Get available positions in a specific room"""
        mid_w = self.width // 2
        mid_h = self.height // 2
        positions = []

        # Define room boundaries
        if room_idx == 0:  # top-left
            x_range = (1, mid_w - 1)
            y_range = (1, mid_h - 1)
        elif room_idx == 1:  # top-right
            x_range = (mid_w + 1, self.width - 2)
            y_range = (1, mid_h - 1)
        elif room_idx == 2:  # bottom-left
            x_range = (1, mid_w - 1)
            y_range = (mid_h + 1, self.height - 2)
        else:  # bottom-right
            x_range = (mid_w + 1, self.width - 2)
            y_range = (mid_h + 1, self.height - 2)

        # Find empty positions in the room
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                if self.grid.get(x, y) is None:
                    positions.append((x, y))

        return positions

    def _generate_memory_questions(self):
        """Generate questions about the placed memory objects"""
        self.memory_questions = []

        if self.memory_test_mode == "object_recall":
            # Questions about object locations
            for i, (obj, pos, color, obj_type) in enumerate(
                self.memory_objects
            ):
                self.memory_questions.append(
                    {
                        "type": "location",
                        "question": f"Where is the {color} {obj_type}?",
                        "correct_answer": pos,
                        "object_idx": i,
                    }
                )

        elif self.memory_test_mode == "color_memory":
            # Questions about object colors
            for i, (obj, pos, color, obj_type) in enumerate(
                self.memory_objects
            ):
                self.memory_questions.append(
                    {
                        "type": "color",
                        "question": f"What color is the {obj_type} at position {pos}?",
                        "correct_answer": color,
                        "object_idx": i,
                    }
                )

        elif self.memory_test_mode == "sequential_memory":
            # Questions about placement sequence
            for i, (obj, pos, color, obj_type) in enumerate(
                self.memory_objects
            ):
                self.memory_questions.append(
                    {
                        "type": "sequence",
                        "question": f"What was the {i+1}th object placed?",
                        "correct_answer": (obj_type, color, pos),
                        "object_idx": i,
                    }
                )

    def _move_steps_from_position(self, start_x, start_y, steps):
        """
        Move exactly 'steps' in a random direction from start position.
        If we hit an obstruction, we stop there.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
        direction = directions[np.random.randint(0, len(directions))]
        dx, dy = direction

        x, y = start_x, start_y
        steps_taken = 0

        while steps_taken < steps:
            nx, ny = x + dx, y + dy

            # Check bounds
            if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                break

            # Check for obstructions
            obj = self.grid.get(nx, ny)
            if obj is not None:
                if isinstance(obj, Wall):
                    break
                if isinstance(obj, Door) and not obj.is_open:
                    break

            # Move to next position
            x, y = nx, ny
            steps_taken += 1

        return (x, y)

    def _eval_memory_performance(self):
        """Evaluate memory performance based on current test mode"""
        memory_metrics = {}

        if self.memory_phase == "question":
            # Evaluate memory recall accuracy
            if self.current_question_idx < len(self.memory_questions):
                question = self.memory_questions[self.current_question_idx]
                memory_metrics['current_question_type'] = question['type']
                memory_metrics['question_idx'] = self.current_question_idx
                memory_metrics['total_questions'] = len(self.memory_questions)

        # Calculate exploration efficiency
        visible_objects = self._get_visible_objects()
        memory_metrics['objects_discovered'] = len(visible_objects)
        memory_metrics['total_objects'] = len(self.memory_objects)
        memory_metrics['exploration_efficiency'] = len(visible_objects) / max(len(self.memory_objects), 1)

        # Calculate memory retention (how many objects agent can still "see" after exploration)
        memory_metrics['memory_retention'] = self._calculate_memory_retention()

        return memory_metrics

    def _calculate_memory_retention(self):
        """Calculate how well the agent retains memory of objects"""
        # This is a simplified metric - in practice, you'd want to test actual recall
        # For now, we'll use the number of objects the agent has seen as a proxy
        visible_objects = self._get_visible_objects()
        return len(visible_objects) / max(len(self.memory_objects), 1)

    def get_memory_test_info(self):
        """Get comprehensive information about the current memory test"""
        return {
            'test_mode': self.memory_test_mode,
            'memory_phase': self.memory_phase,
            'memory_objects': [(pos, color, obj_type) for _, pos, color, obj_type in self.memory_objects],
            'memory_questions': self.memory_questions,
            'current_question_idx': self.current_question_idx,
            'visible_objects': self._get_visible_objects(),
        }

    def _handle_memory_phase(self, action):
        """Handle different phases of memory testing"""
        info = {}

        if self.memory_phase == "exploration":
            # During exploration, track which objects the agent has seen
            info["visible_objects"] = self._get_visible_objects()
            info["exploration_progress"] = len(self.memory_objects) - len(
                [
                    obj
                    for obj in self.memory_objects
                    if obj not in info["visible_objects"]
                ]
            )

            # Check if exploration phase should end (all objects seen or max steps reached)
            if (
                len(info["visible_objects"]) >= len(self.memory_objects)
                or self.step_count >= self.max_steps // 2
            ):
                self.memory_phase = "question"
                info["phase_transition"] = "exploration_to_question"

        elif self.memory_phase == "question":
            # During question phase, evaluate memory responses
            if self.current_question_idx < len(self.memory_questions):
                question = self.memory_questions[self.current_question_idx]
                info["current_question"] = question
                info["question_idx"] = self.current_question_idx

                # For now, we'll evaluate memory in the eval_state method
                # In a real implementation, you might want to handle answers here

                if self.current_question_idx >= len(self.memory_questions) - 1:
                    self.memory_phase = "navigation"
                    info["phase_transition"] = "question_to_navigation"

        elif self.memory_phase == "navigation":
            # During navigation phase, test if agent can navigate to remembered objects
            info["navigation_target"] = self._get_navigation_target()

        return info

    def _get_visible_objects(self):
        """Get visible memory objects using MiniGrid's built-in visibility system"""
        visible = []

        # Check each memory object using built-in visibility methods
        for obj, pos, color, obj_type in self.memory_objects:
            obj_x, obj_y = pos

            # Use built-in method to check if object is visible to agent
            if self._agent_sees(obj_x, obj_y):
                visible.append((obj, pos, color, obj_type))

        return visible

    def _agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = super().gen_obs()

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def _get_navigation_target(self):
        """Get the current navigation target for memory testing"""
        if self.memory_questions and self.current_question_idx < len(
            self.memory_questions
        ):
            question = self.memory_questions[self.current_question_idx]
            if question["type"] == "location":
                return question["correct_answer"]
        return None

    def _get_inward_direction(self, pos):
        """Determine the best direction for agent to face inward (away from walls)"""
        x, y = pos
        mid_w = self.width // 2
        mid_h = self.height // 2

        # For Four Rooms, face toward the center of the room
        if x < mid_w and y < mid_h:  # top-left room
            return 0  # face right (toward center)
        elif x >= mid_w and y < mid_h:  # top-right room  
            return 2  # face left (toward center)
        elif x < mid_w and y >= mid_h:  # bottom-left room
            return 0  # face right (toward center)
        else:  # bottom-right room
            return 2  # face left (toward center)


class TwoRoomsMemoryEnv(FourRoomsMemoryEnv):

    def __init__(
        self,
        world_size: int = 9,
        max_steps: Optional[int] = None,
        see_through_walls: bool = False,
        agent_view_size: int = 4,
        render_mode: Optional[str] = "rgb_array",
        obs_mode: str = "pov",  # "top_down" or "pov"
        tile_size: int = 14,
        seed: Optional[int] = None,
        memory_test_mode: str = "object_recall",  # "navigation", "object_recall", "color_memory", "sequential_memory"
        n_memory_objects: int = 12,  # Number of objects to place for memory tests
        memory_object_types: List[str] = ["ball", "box", "key"],  # Types of objects to place
    ):
        super().__init__(
            world_size=world_size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            obs_mode=obs_mode,
            tile_size=tile_size,
            seed=seed,
            memory_test_mode=memory_test_mode,
            n_memory_objects=n_memory_objects,
            memory_object_types=memory_object_types,
        )
        self.num_rooms = 2

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # outer walls
        self.grid.wall_rect(0, 0, width, height)

        # internal walls to make 4 rooms
        mid_w = width // 2
        mid_h = height // 2
        # self.grid.horz_wall(1, mid_h, width - 2)
        self.grid.vert_wall(mid_w, 1, height - 2)
        self.grid.set(mid_w, mid_h, CustomDoor(color="red"))

        # Place memory objects based on test mode
        if self.memory_test_mode != "navigation":
            self._place_memory_objects()

        # Place goal for navigation tasks
        if self.memory_test_mode == "navigation":
            gx, gy = self.sample_random_pos()
            self.put_obj(Goal(), gx, gy)
            self.goal_pos = (gx, gy)

        # agent spawn
        self.place_agent(init_state=self.init_state)

    def _get_room_positions(self, room_idx):
        """Get available positions in a specific room"""
        mid_w = self.width // 2
        mid_h = self.height // 2
        positions = []

        # Define room boundaries
        if room_idx == 0:  # left
            x_range = (1, mid_w - 1)
            y_range = (1, self.height - 2)
        elif room_idx == 1:  # right
            x_range = (mid_w + 1, self.width - 2)
            y_range = (1, self.height - 2)

        # Find empty positions in the room
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                if self.grid.get(x, y) is None:
                    positions.append((x, y))

        return positions

    def _get_room(self, pos):
        if pos[0] < self.width // 2:
            return 0
        elif pos[0] == self.width // 2:
            return 1
        return 0  # middle hallway

    def _get_visible_objects(self):
        """Get visible memory objects using MiniGrid's built-in visibility system"""
        visible = []
        agent_room = self._get_room(self.agent_pos)

        # Check each memory object using built-in visibility methods
        for obj, pos, color, obj_type in self.memory_objects:
            if self._get_room(pos) == agent_room:
                visible.append((obj, pos, color, obj_type))

        return visible

# -------------------------
# Rollout helpers
# -------------------------
@dataclass
class Trajectory:
    observations: np.ndarray  # [T, H, W, 3]
    actions: np.ndarray  # [T]
    proprio: np.ndarray  # [T, 4]

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
        exploration_actions = list(range(env.action_space.n)) * 5  # repeat each direction 5 times
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
    goal_pos = env.sample_random_pos()
    goal_reached = False
    current_target = goal_pos
    
    while total_steps < max_T:
        current_pos = tuple(env.agent_pos)
        
        # Check if we've reached the current target
        if current_pos == current_target:
            if not goal_reached and current_target == goal_pos:
                goal_reached = True
            
            current_target = env.sample_random_pos()
        
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


def run_scripted_policy_two_rooms(env, max_T, step_and_record):
    """Scripted policy for TwoRoomsMemoryEnv that repeats the exploration sequence 5 times:
    1. Starts at random position in one room
    2. Goes to middle of room using BFS
    3. Goes to door using BFS
    4. Goes to middle of other room using BFS
    5. Returns to original room and original position/direction using BFS
    
    Repeats this sequence 5 times to test memory retention over multiple cycles.
    """
    total_steps = 0
    mid_w = env.width // 2  # 4 for 9x9 grid
    mid_h = env.height // 2  # 4 for 9x9 grid
    door_pos = (mid_w, mid_h)  # (4, 4)
    
    # Store original position and direction
    original_pos = tuple(env.agent_pos)
    original_direction = env.agent_dir
    initial_room = env._get_room(original_pos)
    other_room = 1 - initial_room
    
    def get_room_center(room_idx):
        """Get center position of a room"""
        if room_idx == 0:  # left room
            return (mid_w // 2, mid_h)  # (2, 4)
        else:  # right room
            return (mid_w + mid_w // 2, mid_h)  # (6, 4)
    
    def navigate_to_position(target_pos):
        """Navigate to a specific position using BFS"""
        current_pos = tuple(env.agent_pos)
        path = bfs_shortest_path(env.grid, current_pos, target_pos)
        if path and len(path) > 1:
            planned_actions = plan_actions_from_path(env.agent_dir, path)
            return planned_actions
        return []
    
    def face_direction(target_dir):
        """Face a specific direction"""
        current_dir = env.agent_dir
        diff = (target_dir - current_dir) % 4
        if diff == 1:
            return [1]  # right
        elif diff == 2:
            return [1, 1]  # 180 turn
        elif diff == 3:
            return [0]  # left
        return []  # already facing correct direction
    
    def face_towards_door():
        """Turn to face towards the door"""
        current_pos = tuple(env.agent_pos)
        if current_pos[0] < mid_w:  # in left room, face right
            return face_direction(0)
        else:  # in right room, face left
            return face_direction(2)
    
    def execute_exploration_cycle():
        """Execute one complete exploration cycle"""
        nonlocal total_steps
        
        # Phase 1: Navigate to middle of room
        room_center = get_room_center(initial_room)
        room_actions = navigate_to_position(room_center)
        for action in room_actions:
            step_and_record(action)
            total_steps += 1

        # Phase 2: face door
        face_actions = face_towards_door()
        for action in face_actions:
            step_and_record(action)
            total_steps += 1
        
        # Phase 3: Navigate to door
        door_actions = navigate_to_position(door_pos)
        for action in door_actions:
            step_and_record(action)
            total_steps += 1

        if env._get_room(env.agent_pos) == 0:
            # face left
            step_and_record(0)
            total_steps += 1
        else:
            # face right
            step_and_record(1)
            total_steps += 1

        # Phase 3: Navigate to middle of other room
        other_room_center = get_room_center(other_room)
        room_actions = navigate_to_position(other_room_center)
        for action in room_actions:
            step_and_record(action)
            total_steps += 1

        # navigate to door
        door_actions = navigate_to_position(door_pos)
        for action in door_actions:
            step_and_record(action)
            total_steps += 1

        if env._get_room(env.agent_pos) == 0:
            # face left
            step_and_record(0)
            total_steps += 1
        else:
            # face right
            step_and_record(1)
            total_steps += 1
        
        # Phase 4: Return to original room and position
        return_actions = navigate_to_position(original_pos)
        for action in return_actions:
            step_and_record(action)
            total_steps += 1
        
        # Phase 5: Face original direction
        face_actions = face_direction(original_direction)
        for action in face_actions:
            step_and_record(action)
            total_steps += 1
    
    # Execute the exploration cycle 5 times
    for cycle in range(5):
        execute_exploration_cycle()
    
    # Fill remaining steps with no-op actions if needed
    while total_steps < max_T:
        step_and_record(2)  # forward action (no-op if can't move)
        total_steps += 1
    


def run_bfs_policy_two_rooms(env, max_T, step_and_record, act_random):
    """BFS-based exploration policy for TwoRoomsMemoryEnv that:
    1. Explores current room thoroughly
    2. Goes to door and explores other room
    3. Returns to original room
    """
    total_steps = 0
    mid_w = env.width // 2
    mid_h = env.height // 2
    door_pos = (mid_w, mid_h)
    
    # Track exploration state
    exploration_phase = "initial_room"  # "initial_room", "door", "other_room", "return"
    initial_room = env._get_room(env.agent_pos)
    other_room = 1 - initial_room
    visited_positions = set()
    room_exploration_steps = 0
    max_room_exploration = 15  # Max steps to spend exploring each room
    
    def get_room_center(room_idx):
        """Get center position of a room"""
        if room_idx == 0:  # left room
            return (mid_w // 2, mid_h)
        else:  # right room
            return (mid_w + mid_w // 2, mid_h)
    
    def explore_room_systematically(room_idx, max_steps):
        """Systematically explore a room using BFS to cover different areas"""
        room_center = get_room_center(room_idx)
        current_pos = tuple(env.agent_pos)
        
        # If not in target room, navigate to it first
        if env._get_room(current_pos) != room_idx:
            path = bfs_shortest_path(env.grid, current_pos, room_center)
            if path and len(path) > 1:
                planned_actions = plan_actions_from_path(env.agent_dir, path)
                if planned_actions:
                    return planned_actions[0]
        
        # Generate exploration targets within the room
        exploration_targets = []
        if room_idx == 0:  # left room
            for x in range(1, mid_w - 1, 2):
                for y in range(1, env.height - 2, 2):
                    if env.grid.get(x, y) is None:  # empty cell
                        exploration_targets.append((x, y))
        else:  # right room
            for x in range(mid_w + 1, env.width - 2, 2):
                for y in range(1, env.height - 2, 2):
                    if env.grid.get(x, y) is None:  # empty cell
                        exploration_targets.append((x, y))
        
        # Find closest unexplored target
        current_pos = tuple(env.agent_pos)
        best_target = None
        best_distance = float('inf')
        
        for target in exploration_targets:
            if target not in visited_positions:
                path = bfs_shortest_path(env.grid, current_pos, target)
                if path and len(path) < best_distance:
                    best_target = target
                    best_distance = len(path)
        
        if best_target:
            path = bfs_shortest_path(env.grid, current_pos, best_target)
            if path and len(path) > 1:
                planned_actions = plan_actions_from_path(env.agent_dir, path)
                if planned_actions:
                    return planned_actions[0]
        
        # Fallback: random action
        return act_random()
    
    while total_steps < max_T:
        current_pos = tuple(env.agent_pos)
        visited_positions.add(current_pos)
        
        if exploration_phase == "initial_room":
            # Explore the initial room thoroughly
            if room_exploration_steps < max_room_exploration and env._get_room(current_pos) == initial_room:
                action = explore_room_systematically(initial_room, max_room_exploration - room_exploration_steps)
                step_and_record(action)
                total_steps += 1
                room_exploration_steps += 1
            else:
                # Move to door
                path = bfs_shortest_path(env.grid, current_pos, door_pos)
                if path and len(path) > 1:
                    planned_actions = plan_actions_from_path(env.agent_dir, path)
                    if planned_actions:
                        action = planned_actions[0]
                        step_and_record(action)
                        total_steps += 1
                        if tuple(env.agent_pos) == door_pos:
                            exploration_phase = "other_room"
                            room_exploration_steps = 0
                    else:
                        step_and_record(act_random())
                        total_steps += 1
                else:
                    step_and_record(act_random())
                    total_steps += 1
        
        elif exploration_phase == "other_room":
            # Explore the other room
            if room_exploration_steps < max_room_exploration and env._get_room(current_pos) == other_room:
                action = explore_room_systematically(other_room, max_room_exploration - room_exploration_steps)
                step_and_record(action)
                total_steps += 1
                room_exploration_steps += 1
            else:
                # Move back to door
                path = bfs_shortest_path(env.grid, current_pos, door_pos)
                if path and len(path) > 1:
                    planned_actions = plan_actions_from_path(env.agent_dir, path)
                    if planned_actions:
                        action = planned_actions[0]
                        step_and_record(action)
                        total_steps += 1
                        if tuple(env.agent_pos) == door_pos:
                            exploration_phase = "return"
                    else:
                        step_and_record(act_random())
                        total_steps += 1
                else:
                    step_and_record(act_random())
                    total_steps += 1
        
        elif exploration_phase == "return":
            # Return to initial room and continue exploring
            if env._get_room(current_pos) == initial_room:
                exploration_phase = "initial_room"
                room_exploration_steps = 0
            else:
                # Navigate back to initial room
                initial_room_center = get_room_center(initial_room)
                path = bfs_shortest_path(env.grid, current_pos, initial_room_center)
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
    - 'scripted': Scripted policy for TwoRoomsMemoryEnv (looks both directions, goes to door, explores other room, returns)
    """
    obs_list, act_list, proprio_list = [], [], []

    obs, _ = env.reset()
    obs_list.append(obs['visual'])
    proprio_list.append(obs['proprio'])
    act_list.append(0)
    max_T = max_steps or env.max_steps

    def step_and_record(action):
        """Helper to step environment and record trajectory data"""
        obs, _, _, _, _ = env.step(action)
        obs_list.append(obs['visual'])
        proprio_list.append(obs['proprio'])
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
        elif isinstance(env, TwoRoomsMemoryEnv):
            run_bfs_policy_two_rooms(env, max_T, step_and_record, act_random)
        else:
            # Unknown environment, fall back to random
            raise ValueError(f"Unknown environment: {type(env)}")
    elif policy == "scripted":
        if isinstance(env, TwoRoomsMemoryEnv):
            run_scripted_policy_two_rooms(env, max_T, step_and_record)
        else:
            raise ValueError(f"Scripted policy only supported for TwoRoomsMemoryEnv, got {type(env)}")
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    # cap the length of the trajectory to max_T
    obs_list = obs_list[:max_T]
    act_list = act_list[:max_T]
    proprio_list = proprio_list[:max_T]

    obs_list = np.stack(obs_list, axis=0) # [max_T, H, W, 3]
    # obs_list = {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0].keys()}
    act_list = np.array(act_list)
    proprio_list = np.stack(proprio_list, axis=0) # [max_T, 4]

    traj = Trajectory(
        observations=obs_list,
        actions=act_list,
        proprio=proprio_list,
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
    # observations = {k: np.stack([t.observations[k] for t in trajectories], axis=0) for k in trajectories[0].observations.keys()}
    actions = np.stack([t.actions for t in trajectories])           # (N, T)
    proprio = np.stack([t.proprio for t in trajectories])           # (N, T, 4)
    
    # Save as separate NPY files
    obs_path = os.path.join(out_dir, f"observations_{chunk_idx:04d}.npy")
    act_path = os.path.join(out_dir, f"actions_{chunk_idx:04d}.npy")
    proprio_path = os.path.join(out_dir, f"proprio_{chunk_idx:04d}.npy")
    
    np.save(obs_path, observations)
    np.save(act_path, actions)
    np.save(proprio_path, proprio)
    return obs_path, act_path, proprio_path


def make_four_rooms(world_size=17, obs_mode="top_down", tile_size=14, agent_view_size=None, 
                   memory_test_mode="object_recall", n_memory_objects=16, memory_object_types=None):
    return FourRoomsMemoryEnv(
        world_size=world_size, 
        obs_mode=obs_mode, 
        tile_size=tile_size, 
        agent_view_size=agent_view_size,
        memory_test_mode=memory_test_mode,
        n_memory_objects=n_memory_objects,
        memory_object_types=memory_object_types
    )


def make_two_rooms(world_size=9, obs_mode="pov", tile_size=14, agent_view_size=4, 
                   memory_test_mode="object_recall", n_memory_objects=12, memory_object_types=["ball", "box", "key"]):
    return TwoRoomsMemoryEnv(
        world_size=world_size, 
        obs_mode=obs_mode, 
        tile_size=tile_size, 
        agent_view_size=agent_view_size,
        memory_test_mode=memory_test_mode,
        n_memory_objects=n_memory_objects,
        memory_object_types=memory_object_types
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
        "--env", choices=["four_rooms", "two_rooms"], required=True
    )
    g.add_argument("--episodes", type=int, default=1000)
    g.add_argument("--output-dir", type=str, default="minigrid_env")
    g.add_argument("--policy", choices=["random", "bfs", "explore", "scripted"], default="random")
    g.add_argument("--max-steps", type=int, default=100)
    g.add_argument("--episodes-per-chunk", type=int, default=100)
    # Memory testing parameters
    g.add_argument("--memory-test-mode", choices=["navigation", "object_recall", "color_memory", "sequential_memory"], 
                   default="object_recall", help="Memory test mode for four_rooms environment")
    g.add_argument("--n-memory-objects", type=int, default=16, help="Number of memory objects to place")
    g.add_argument("--memory-object-types", nargs="+", default=["ball", "box", "key"], 
                   help="Types of memory objects to place")
    
    args = parser.parse_args()

    # Regular dataset generation or evaluation
    if args.env == "four_rooms":
        ctor = lambda: make_four_rooms(
            memory_test_mode=args.memory_test_mode,
            n_memory_objects=args.n_memory_objects,
            memory_object_types=args.memory_object_types
        )
    elif args.env == "two_rooms":
        ctor = lambda: make_two_rooms(
            memory_test_mode=args.memory_test_mode,
            n_memory_objects=args.n_memory_objects,
            memory_object_types=args.memory_object_types
        )
    
    dataset_dir = os.environ["DATASET_DIR"]
    assert dataset_dir is not None, "DATASET_DIR must be set"
    
    # Include memory test mode in output path
    memory_suffix = f"_{args.memory_test_mode}"
    output_path = os.path.join(dataset_dir, args.output_dir, f"{args.env}_{args.policy}{memory_suffix}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize chunking variables
    current_chunk = []
    chunk_idx = 0
    total_episodes = 0
    
    # Determine if we're generating memory trajectories
    is_memory_test = args.memory_test_mode != 'navigation'
    
    for episode_idx in tqdm(range(args.episodes), desc="Generating"):
        env = ctor()
        
        traj = run_episode(env, max_steps=args.max_steps, policy=args.policy)
        
        current_chunk.append(traj)
        env.close()
        total_episodes += 1
        
        # Save chunk when it reaches the target size
        if len(current_chunk) >= args.episodes_per_chunk:
            paths = save_trajectories_npy(current_chunk, output_path, chunk_idx)
        
            print(f"Saved chunk {chunk_idx} with {len(current_chunk)} episodes")
            current_chunk = []
            chunk_idx += 1
    
    # Save final partial chunk if it has any episodes
    if current_chunk:
        paths = save_trajectories_npy(current_chunk, output_path, chunk_idx)

        print(f"Saved final chunk {chunk_idx} with {len(current_chunk)} episodes")
        chunk_idx += 1
    
    # Create index file
    index = {
        'total_episodes': total_episodes,
        'episodes_per_chunk': args.episodes_per_chunk,
        'n_chunks': chunk_idx,
        'policy': args.policy,
        'max_steps': args.max_steps,
        'output_dir': args.output_dir,
        'env': args.env,
    }
    
    # Add memory-specific metadata
    if is_memory_test:
        index.update({
            'memory_test_mode': args.memory_test_mode,
            'n_memory_objects': args.n_memory_objects,
            'memory_object_types': args.memory_object_types,
        })
    
    index_path = os.path.join(output_path, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Saved {total_episodes} episodes in {chunk_idx} chunks to {output_path}")


if __name__ == "__main__":
    main()
