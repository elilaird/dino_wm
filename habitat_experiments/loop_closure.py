"""
Loop-closure evaluation runner for Habitat-Lab.

What this does:
- Loads a Habitat environment (config path provided).
- Builds deterministic loop trajectories (start -> far waypoint -> back to start) using ShortestPathFollower.
- Replays the EXACT same action sequence for any model variant.
- Logs GT poses, per-step sink metrics (if the model provides token embeddings & attention),
  and optional features for later analysis.

You need:
- habitat-lab + habitat-sim installed
- a dataset config that points to valid scenes (e.g., HM3D/Gibson/Replica)

Fill in your model in the ModelAdapter class (end of file).
"""

import os
import json
import math
import time
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import habitat
from habitat import Env, RLEnv
from habitat.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

# ----------------------------
# Utility: reproducibility
# ----------------------------
def set_global_seed(seed: int):
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------
# Loop Builder
# ----------------------------
@dataclass
class LoopSpec:
    start_pos: np.ndarray
    start_rot: np.ndarray  # quaternion (x, y, z, w)
    waypoints_xyz: List[np.ndarray]
    actions: List[str]  # ['MOVE_FORWARD', 'TURN_LEFT', ...]
    approx_length_m: float


class LoopBuilder:
    """
    Builds closed loops using geodesic shortest paths.

    Strategy:
      - Sample a valid start s0.
      - Sample candidate waypoint w1 until geodesic_distance(s0, w1) ~ target_len/2.
      - Build path s0->w1 and w1->s0 via ShortestPathFollower (discrete actions).
    """

    def __init__(
        self,
        sim,
        goal_radius: float = 0.20,
        forward_step_size: float = 0.25,
        turn_angle: float = 10.0,
        max_samples: int = 2000,
    ):
        self.sim = sim
        self.goal_radius = goal_radius
        self.forward_step_size = forward_step_size
        self.turn_angle = turn_angle
        self.max_samples = max_samples

        # Action space assumptions – ensure your config matches these names
        self.ACTION_MOVE = "MOVE_FORWARD"
        self.ACTION_TURN_LEFT = "TURN_LEFT"
        self.ACTION_TURN_RIGHT = "TURN_RIGHT"

        # Follower for path-to-actions
        self.follower = ShortestPathFollower(
            self.sim, goal_radius=self.goal_radius, return_one_hot=False
        )

    def _geodesic_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        try:
            d = self.sim.geodesic_distance(a, b)
        except Exception:
            d = self.sim.pathfinder.get_geodesic_distance(a, b)
        return float(d)

    def _sample_valid_point(self) -> np.ndarray:
        return self.sim.pathfinder.get_random_navigable_point()

    def _agent_state(self) -> Tuple[np.ndarray, np.ndarray]:
        st = self.sim.get_agent_state()
        return np.array(st.position), np.array([st.rotation.x, st.rotation.y, st.rotation.z, st.rotation.w])

    def _reset_to(self, pos: np.ndarray, rot_xyzw: np.ndarray):
        state = habitat_sim.AgentState()  # type: ignore
        state.position = pos
        quat = habitat_sim.utils.quat_from_xyzw(rot_xyzw)  # type: ignore
        state.rotation = quat
        self.sim.set_agent_state(state.position, state.rotation)

    def _path_to_actions(self, start_pos: np.ndarray, start_rot: np.ndarray, goal_pos: np.ndarray) -> Tuple[List[str], List[np.ndarray]]:
        # Reset follower’s internal state by resetting agent
        import habitat_sim  # lazy import to avoid issues if not needed elsewhere
        self._reset_to(start_pos, start_rot)

        actions: List[str] = []
        waypoints: List[np.ndarray] = []

        # ask follower for next action until at goal
        last = None
        max_steps = 20000  # safety
        for _ in range(max_steps):
            act = self.follower.get_next_action(goal_pos)
            if act is None:
                break
            actions.append(act.name if hasattr(act, "name") else str(act))
            # Step sim to get the new position
            self.sim.step(act)
            p, _ = self._agent_state()
            waypoints.append(p)
            last = p

            if np.linalg.norm(last - goal_pos) <= self.goal_radius * 1.5:
                break

        return actions, waypoints

    def build_loop(
        self,
        target_length_m: float,
        heading_deg: Optional[float] = None,
        tries: int = 200,
    ) -> LoopSpec:
        import habitat_sim

        # Sample s0
        start = self._sample_valid_point()

        # Pick a heading if requested
        if heading_deg is None:
            heading_deg = 0.0
        yaw = math.radians(heading_deg)
        # Convert yaw to quaternion (xyzw)
        quat = habitat_sim.utils.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))  # y-up
        start_rot_xyzw = np.array([quat.x, quat.y, quat.z, quat.w])

        # Find midpoint with ~ L/2 geodesic distance
        half = max(0.1, target_length_m / 2.0)
        candidate = None
        for _ in range(self.max_samples):
            p = self._sample_valid_point()
            d = self._geodesic_dist(start, p)
            if np.isfinite(d) and abs(d - half) <= 0.10 * half:
                candidate = p
                break

        if candidate is None:
            # Fall back: just pick a farthest-of-tries candidate
            best_d = -1.0
            best_p = None
            for _ in range(tries):
                p = self._sample_valid_point()
                d = self._geodesic_dist(start, p)
                if d > best_d and np.isfinite(d):
                    best_d = d
                    best_p = p
            candidate = best_p if best_p is not None else self._sample_valid_point()

        # Build s0 -> w1
        acts_fw, wp_fw = self._path_to_actions(start, start_rot_xyzw, candidate)
        # Build w1 -> s0 (reset follower by starting at candidate)
        acts_bw, wp_bw = self._path_to_actions(candidate, start_rot_xyzw, start)

        actions = acts_fw + acts_bw
        waypoints = [np.array(start)] + wp_fw + [np.array(candidate)] + wp_bw + [np.array(start)]
        approx_len = float(self._geodesic_dist(start, candidate) + self._geodesic_dist(candidate, start))

        return LoopSpec(
            start_pos=np.array(start),
            start_rot=start_rot_xyzw,
            waypoints_xyz=waypoints,
            actions=actions,
            approx_length_m=approx_len,
        )


# ----------------------------
# Logging
# ----------------------------
class RunLogger:
    """
    Writes:
      - meta.json: config for the run (scene, loop length, seed, etc.)
      - steps.jsonl: per-step metrics (pose, sink stats, etc.)
      - arrays/ : optional numpy arrays (features, attention) per step
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.arr_dir = os.path.join(out_dir, "arrays")
        os.makedirs(self.arr_dir, exist_ok=True)
        self.meta_path = os.path.join(out_dir, "meta.json")
        self.steps_path = os.path.join(out_dir, "steps.jsonl")
        self.step_fp = open(self.steps_path, "w", buffering=1)

    def write_meta(self, meta: Dict):
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_step(
        self,
        step_idx: int,
        pose: Dict[str, List[float]],
        sink_rate: Optional[float] = None,
        attn_entropy: Optional[float] = None,
        anchor_mass: Optional[float] = None,
        extra: Optional[Dict] = None,
    ):
        row = {
            "step": step_idx,
            "pose": pose,
            "sink_rate": sink_rate,
            "attn_entropy": attn_entropy,
            "anchor_mass": anchor_mass,
        }
        if extra:
            row.update(extra)
        self.step_fp.write(json.dumps(row) + "\n")

    def save_array(self, name: str, arr: np.ndarray, step_idx: int):
        np.save(os.path.join(self.arr_dir, f"{name}_{step_idx:06d}.npy"), arr)

    def close(self):
        try:
            self.step_fp.close()
        except Exception:
            pass


# ----------------------------
# Sink metrics helpers
# ----------------------------
def high_norm_outlier_rate(token_embeddings: np.ndarray, anchor_mask: Optional[np.ndarray] = None, kappa: float = 3.0) -> float:
    """
    token_embeddings: (L, D) OR (H, L, D) – we'll flatten heads if present.
    anchor_mask: (L,) boolean mask of anchor positions to ignore.
    """
    X = token_embeddings
    if X.ndim == 3:  # (H, L, D)
        X = X.reshape(-1, X.shape[-2], X.shape[-1]).mean(axis=0)  # average heads
    norms = np.linalg.norm(X, axis=-1)  # (L,)

    if anchor_mask is not None and anchor_mask.shape[0] == norms.shape[0]:
        norms = norms[~anchor_mask]

    mu = norms.mean()
    sigma = norms.std() + 1e-8
    outlier = (norms > (mu + kappa * sigma)).astype(np.float32)
    return float(outlier.mean())


def attention_entropy_and_anchor_mass(attn_probs: np.ndarray, anchor_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    attn_probs: (H, L_q, L_k) or (L_q, L_k); probabilities per head.
    Returns:
      - mean entropy over non-anchor keys (averaged across queries/heads)
      - mean attention mass on anchor keys (averaged across queries/heads)
    """
    A = attn_probs
    if A.ndim == 2:
        A = A[None, ...]  # (1, Lq, Lk)
    H, Lq, Lk = A.shape
    # entropy over keys
    eps = 1e-12
    ent = -(A * np.log(A + eps)).sum(axis=-1)  # (H, Lq)
    mean_ent = float(ent.mean())

    anchor_mass = None
    if anchor_mask is not None and anchor_mask.shape[0] == Lk:
        mass = (A * anchor_mask.astype(np.float32)[None, None, :]).sum(axis=-1)  # (H, Lq)
        anchor_mass = float(mass.mean())
    else:
        anchor_mass = float("nan")
    return mean_ent, anchor_mass


# ----------------------------
# Execution (action replay)
# ----------------------------
def run_loop_with_actions(
    env: Env,
    loop: LoopSpec,
    model: "ModelAdapter",
    out_dir: str,
    save_arrays_every: int = 0,
    seed: int = 0,
):
    logger = RunLogger(out_dir)
    logger.write_meta({
        "seed": seed,
        "scene_id": env.habitat_config.habitat.simulator.scene,
        "approx_length_m": loop.approx_length_m,
        "n_actions": len(loop.actions),
        "start_pos": loop.start_pos.tolist(),
        "start_rot_xyzw": loop.start_rot.tolist(),
    })

    # Reset agent to start
    # NOTE: habitat-lab wraps habitat-sim; we use sim control for precise placement
    sim = env.sim
    import habitat_sim
    start_state = habitat_sim.AgentState()
    start_state.position = loop.start_pos
    start_state.rotation = habitat_sim.utils.quat_from_xyzw(loop.start_rot)
    sim.set_agent_state(start_state.position, start_state.rotation)
    env._dataset = None  # avoid episode stepping; we control sim directly

    # Roll
    for t, act_name in enumerate(loop.actions):
        # Step the sim (action replay)
        obs = env.step(act_name)

        # Get GT pose
        st = sim.get_agent_state()
        pose = {
            "pos": [float(x) for x in st.position.tolist()],
            "rot_xyzw": [float(st.rotation.x), float(st.rotation.y), float(st.rotation.z), float(st.rotation.w)],
        }

        # Ask model for metrics (features/attention/sinks)
        model_out = model.step(obs, act_name)

        sink_rate, attn_ent, anchor_mass = None, None, None
        if model_out is not None:
            # Expect dict with optional keys:
            #   'token_embeddings': np.ndarray (L,D) or (H,L,D)
            #   'attention_probs': np.ndarray (H,Lq,Lk) or (Lq,Lk)
            #   'anchor_mask': np.ndarray (L,) boolean
            tm = model_out.get("token_embeddings")
            amask = model_out.get("anchor_mask")
            attn = model_out.get("attention_probs")

            if tm is not None:
                sink_rate = high_norm_outlier_rate(tm, amask)

            if attn is not None:
                ent, am = attention_entropy_and_anchor_mass(attn, amask)
                attn_ent, anchor_mass = ent, am

            # Optionally save arrays
            if save_arrays_every and (t % save_arrays_every == 0):
                if tm is not None:
                    logger.save_array("token_embeddings", np.asarray(tm), t)
                if attn is not None:
                    logger.save_array("attention_probs", np.asarray(attn), t)
                if amask is not None:
                    logger.save_array("anchor_mask", np.asarray(amask), t)

        logger.log_step(
            step_idx=t,
            pose=pose,
            sink_rate=sink_rate,
            attn_entropy=attn_ent,
            anchor_mass=anchor_mass,
        )

    logger.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to Habitat config (.yaml)")
    parser.add_argument("--scene", type=str, default=None, help="Override scene id")
    parser.add_argument("--out", type=str, required=True, help="Output dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop_m", type=float, default=100.0, help="Target loop length (meters)")
    parser.add_argument("--heading_deg", type=float, default=0.0)
    parser.add_argument("--save_arrays_every", type=int, default=0, help="0=off, else save arrays every k steps")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_global_seed(args.seed)

    # Load config/env
    cfg = get_config(args.config)
    if args.scene is not None:
        cfg.habitat.simulator.scene = args.scene

    # Make sure actions match LoopBuilder assumptions:
    # The default Habitat PointNav actions are usually:
    #   MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    # Ensure your config uses these names (or edit LoopBuilder).
    with habitat.Env(config=cfg) as env:
        # Build loop
        lb = LoopBuilder(env.sim)
        loop = lb.build_loop(target_length_m=args.loop_m, heading_deg=args.heading_deg)

        # Plug in your model here:
        model = ModelAdapter()  # replace with your implementation

        # Run + log
        run_loop_with_actions(
            env=env,
            loop=loop,
            model=model,
            out_dir=args.out,
            save_arrays_every=args.save_arrays_every,
            seed=args.seed,
        )

    print(f"Done. Logs at: {args.out}")


# ----------------------------
# Model Adapter (YOU implement this)
# ----------------------------
class ModelAdapter:
    """
    Replace this with your world model wrapper.

    Contract:
      step(obs, action_str) -> Dict or None
        Return a dict with any of these keys (all optional):
          - 'token_embeddings': np.ndarray (L,D) or (H,L,D)
            (e.g., last-layer token states; include anchors + non-anchors in sequence order)
          - 'attention_probs': np.ndarray (H,Lq,Lk) or (Lq,Lk)
            (e.g., average over heads or per-head softmax weights for the last attention)
          - 'anchor_mask': np.ndarray (L,) boolean (True for anchor positions)
            (registers/persistent tokens + Titans h_t position)
        If you return None, we’ll just log pose.

    Notes:
      - You control how to tokenize rgb/depth obs into your model’s inputs.
      - For Titans variants, set the correct anchor_mask (prefix anchors and, if MAC, the h_t slot).
    """
    def __init__(self):
        # TODO: load checkpoints, tokenizers, etc.
        self._dummy = True

    def step(self, obs: Dict, action_str: str) -> Optional[Dict]:
        # EXAMPLE (dummy): produce fake arrays matching required shapes
        # Replace with your actual model forward pass.
        if self._dummy:
            L = 196 + 1 + 4   # e.g., 14x14 = 196 visual tokens + 1 action token + 4 anchors
            D = 512
            H = 8
            token_embeddings = np.random.randn(L, D).astype(np.float32) * 0.01
            attention_probs = np.random.dirichlet(np.ones(L), size=(H, L)).astype(np.float32)  # (H,L,L)
            anchor_mask = np.zeros((L,), dtype=bool)
            # suppose first 4 are anchors:
            anchor_mask[:4] = True
            return {
                "token_embeddings": token_embeddings,
                "attention_probs": attention_probs,
                "anchor_mask": anchor_mask,
            }
        return None


if __name__ == "__main__":
    # lazy import to avoid issues at import time if habitat_sim not installed
    try:
        import habitat_sim  # noqa: F401
    except Exception:
        pass
    main()

