from gymnasium.envs.registration import register
from .pointmaze import U_MAZE
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
    disable_env_checker=True,
)
register(
    id='point_maze',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
    disable_env_checker=True,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
    disable_env_checker=True,
)

register(
    id="four_rooms_random",
    entry_point="env.minigrid.minigrid_env:FourRoomsMemoryEnv",
    max_episode_steps=500,
    reward_threshold=1.0,
    disable_env_checker=True,
    kwargs={
        'world_size': 17,
        'obs_mode': "top_down",
        'tile_size': 14,
        'agent_view_size': 7,
    }
)

register(
    id="four_rooms_explore",
    entry_point="env.minigrid.minigrid_env:FourRoomsMemoryEnv",
    max_episode_steps=500,
    reward_threshold=1.0,
    disable_env_checker=True,
    kwargs={
        'world_size': 17,
        'obs_mode': "top_down",
        'tile_size': 14,
        'agent_view_size': 7,
    }
)

register(
    id="four_rooms_bfs",
    entry_point="env.minigrid.minigrid_env:FourRoomsMemoryEnv",
    max_episode_steps=500,
    reward_threshold=1.0,
    disable_env_checker=True,
    kwargs={
        'world_size': 17,
        'obs_mode': "top_down",
        'tile_size': 14,
        'agent_view_size': 7,
    }
)