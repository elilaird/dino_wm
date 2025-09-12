from memory_maze import tasks
from memory_maze.gymnasium_wrapper import GymnasiumWrapper
import numpy as np

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

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


class MemMazeWrapper(GymnasiumWrapper):
    def __init__(self, **kwargs):
        # Create the underlying memory_maze environment
        env = tasks.memory_maze_two_rooms_3x7_fixed_layout_random_goals(**kwargs)
        super().__init__(env)
        self.action_dim = self.action_space.n  # or shape[0] for continuous
        self.maze_xy_scale = None
        self.maze_width = None
        self.maze_height = None
        self.center_ji = None
        
    def sample_random_init_goal_states(self, seed):
        """
        Sample random initial and goal positions in the maze.
        Returns positions that can be used with prepare() method.
        """
        np.random.seed(seed)
        
        # Reset environment to get current target positions in observation format
        obs, _ = super().reset(seed=seed)
        init_pos = obs['agent_pos']
        goal_pos = obs['targets_pos'][np.random.randint(len(obs['targets_pos'] ))]
        
        return init_pos, goal_pos
        
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }
    
    def set_maze_params(self):
        self.maze_xy_scale = self.env._task._maze_arena.xy_scale
        self.maze_width = self.env._task._maze_arena.maze.width
        self.maze_height = self.env._task._maze_arena.maze.height
        self.center_ji = np.array([self.maze_width - 2.0, self.maze_height - 2.0]) / 2.0
    
    def grid_to_world(self, grid_pos):
        return (grid_pos - self.center_ji) * self.maze_xy_scale
    
    def world_to_grid(self, world_pos):
        return world_pos / self.maze_xy_scale + self.center_ji
        
    def prepare(self, seed, init_state):
        """
        Set the agent to the specified initial state and return observation.
        init_state: [x, y, z] position in the maze
        """
        if self.maze_xy_scale is None or self.maze_width is None or self.maze_height is None or self.center_ji is None:
            self.set_maze_params()

        # super().reset(seed=seed)
        world_init_state = self.grid_to_world(init_state)
        self.env._task._walker.shift_pose(
            self.env.physics, 
            [world_init_state[0], world_init_state[1], 0.1], 
            rotate_velocity=True
        )
        obs, _, _, _, info = self.step(0)# no-op action
        state = info["state"]
        
        return obs, state

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
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
        
    def update_env(self, env_info):
        # Update environment configuration
        pass

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = {}
        info['state'] = obs["agent_pos"]
        proprio = np.concatenate([obs["agent_pos"], obs["agent_dir"]])

        obs = {
            "visual": obs["image"],
            "proprio": proprio
        }
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        info['state'] = obs["agent_pos"]
        obs = {
            "visual": obs["image"],
            "proprio": np.concatenate([obs["agent_pos"], obs["agent_dir"]])
        }
        return obs, info

    