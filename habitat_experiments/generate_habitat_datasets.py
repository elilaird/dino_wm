# tools/generate_memory_datasets.py
import os, json, gzip, uuid, argparse, numpy as np
from tqdm import trange
import habitat
from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry

# Ensure custom actions are registered (e.g., Teleport)
import importlib

try:
    importlib.import_module("habitat_experiments.extensions.teleport_action")
except Exception:
    pass


def _obs_to_numpy(obs):
    out = {}
    for k, v in obs.items():
        if hasattr(v, "shape"):
            out[k] = np.array(v)
    return out


def _get_goal_context(obs, cfg):
    # Supports ImageNav/DelayedRecall via GOAL_SENSOR_UUID
    goal_uuid = cfg.TASK.get("GOAL_SENSOR_UUID", None)
    if goal_uuid and goal_uuid in obs:
        return np.array(obs[goal_uuid])
    return None


def random_policy(env):
    return env.action_space.sample()


def forward_bias_policy(env):
    # small heuristic: try forward often, with some turning
    import random

    return np.random.choice(
        [0, 1, 2, 3], p=[0.05, 0.55, 0.20, 0.20]
    )  # STOP, FWD, L, R


def run_episodes(exp_config, out_path, episodes=200, policy="forward_bias"):
    cfg = habitat.get_config(exp_config)
    # Optional live tweaks
    with read_write(cfg):
        if "MAX_EPISODE_STEPS" in cfg.ENVIRONMENT:
            cfg.ENVIRONMENT.MAX_EPISODE_STEPS = int(
                cfg.ENVIRONMENT.MAX_EPISODE_STEPS
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    trajs = []

    with habitat.Env(config=cfg) as env:
        for _ in trange(episodes, desc="Collecting"):
            obs = env.reset()
            obs_np = _obs_to_numpy(obs)
            goal_ctx = _get_goal_context(obs, cfg)

            frames, acts, rews, dones, infos = [obs_np], [], [], [], []
            total_r = 0.0
            done = False
            steps = 0

            while not done:
                a = (
                    forward_bias_policy(env)
                    if policy == "forward_bias"
                    else random_policy(env)
                )
                obs, r, done, info = env.step(a)
                frames.append(_obs_to_numpy(obs))
                acts.append(a)
                rews.append(float(r))
                dones.append(bool(done))
                infos.append(info)
                total_r += float(r)
                steps += 1

            trajs.append(
                {
                    "frames": frames,  # dicts: {"rgb": [H,W,3], "depth": [H,W,1], ...}
                    "actions": np.array(acts, dtype=np.int64),
                    "rewards": np.array(rews, dtype=np.float32),
                    "dones": np.array(dones, dtype=np.bool_),
                    "infos": json.dumps(infos),
                    "goal_context": (
                        goal_ctx
                        if goal_ctx is not None
                        else np.array([], dtype=np.uint8)
                    ),
                    "task_type": cfg.TASK.TYPE,
                    "dataset_type": cfg.DATASET.TYPE,
                    "episode_id": str(uuid.uuid4()),
                }
            )

    # save as npz (lists of variable-length episodes)
    pack = {
        "frames": [t["frames"] for t in trajs],
        "actions": [t["actions"] for t in trajs],
        "rewards": [t["rewards"] for t in trajs],
        "dones": [t["dones"] for t in trajs],
        "infos": [t["infos"] for t in trajs],
        "goal_context": [t["goal_context"] for t in trajs],
        "task_type": [t["task_type"] for t in trajs],
        "dataset_type": [t["dataset_type"] for t in trajs],
        "episode_id": [t["episode_id"] for t in trajs],
    }
    np.savez_compressed(out_path, **pack)
    print(f"Saved {len(trajs)} episodes to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML (e.g., configs/memory/image_goal_nav.yaml)",
    )
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument(
        "--policy", choices=["random", "forward_bias"], default="forward_bias"
    )
    args = ap.parse_args()
    run_episodes(args.config, args.out, args.episodes, args.policy)
