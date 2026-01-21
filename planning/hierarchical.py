import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device
from planning.objectives import create_objective_fn
import hydra


class HierarchicalPlanner(BasePlanner):
    """
    Hierarchical planner implementing coarse-to-fine MPC with varying temporal resolutions.
    
    Architecture:
    - Coarse Planner: High frameskip, long horizon for strategic planning
    - Fine Planner: Low frameskip, short horizon for tactical refinement
    - Coupling: Fine planner tracks coarse trajectory via tracking cost
    """

    def __init__(
        self,
        coarse_planner,
        fine_planner,
        coarse_frameskip,
        fine_frameskip,
        tracking_weight,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        upsample_method="repeat",
        logging_prefix="hierarchical",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )

        self.coarse_frameskip = coarse_frameskip
        self.fine_frameskip = fine_frameskip
        self.tracking_weight = tracking_weight
        self.upsample_method = upsample_method
        self.logging_prefix = logging_prefix

        # Initialize coarse planner
        coarse_planner["_target_"] = coarse_planner["target"]
        coarse_planner["wm"] = wm
        coarse_planner["action_dim"] = action_dim
        coarse_planner["objective_fn"] = objective_fn
        coarse_planner["preprocessor"] = preprocessor
        coarse_planner["evaluator"] = None  # Disable intermediate evaluation
        coarse_planner["wandb_run"] = wandb_run
        coarse_planner["log_filename"] = None
        coarse_planner["logging_prefix"] = f"{logging_prefix}_coarse"

        self.coarse_planner = hydra.utils.instantiate(coarse_planner)

        # Initialize fine planner
        fine_planner["_target_"] = fine_planner["target"]
        fine_planner["wm"] = wm
        fine_planner["action_dim"] = action_dim
        fine_planner["objective_fn"] = self._create_tracking_objective_fn()
        fine_planner["preprocessor"] = preprocessor
        fine_planner["evaluator"] = evaluator
        fine_planner["wandb_run"] = wandb_run
        fine_planner["log_filename"] = log_filename
        fine_planner["logging_prefix"] = f"{logging_prefix}_fine"

        self.fine_planner = hydra.utils.instantiate(fine_planner)

    def _create_tracking_objective_fn(self):
        """Create objective function that combines original goal with tracking cost."""
        base_objective = self.objective_fn

        def tracking_objective_fn(z_obs_pred, z_obs_tgt):
            """
            Combined objective: base goal cost + tracking cost to coarse trajectory
            
            Args:
                z_obs_pred: dict with predicted observations
                z_obs_tgt: dict with target observations (includes tracking targets)
            """
            # Extract goal target (last element in time dimension)
            z_obs_goal = {
                key: arr[:, -1:] for key, arr in z_obs_tgt.items()
            }

            # Extract tracking targets (all elements except last)
            z_obs_track = {
                key: arr[:, :-1] for key, arr in z_obs_tgt.items()
            }

            # Base goal cost (terminal cost)
            goal_cost = base_objective(z_obs_pred, z_obs_goal)

            # Tracking cost (path following)
            if z_obs_pred["visual"].shape[1] > 1:
                track_cost = base_objective(
                    {key: arr[:, :-1] for key, arr in z_obs_pred.items()},
                    z_obs_track
                )
                total_cost = goal_cost + self.tracking_weight * track_cost
            else:
                # No intermediate tracking points
                total_cost = goal_cost

            return total_cost

        return tracking_objective_fn

    def _upsample_actions(self, coarse_actions, coarse_frameskip, fine_frameskip, target_horizon):
        """
        Upsample coarse actions to fine resolution.
        
        Args:
            coarse_actions: (B, T_coarse, action_dim) tensor
            coarse_frameskip: int
            fine_frameskip: int
            target_horizon: target fine horizon
            
        Returns:
            upsampled_actions: (B, T_fine, action_dim) tensor
        """
        batch_size, coarse_horizon, action_dim = coarse_actions.shape

        if self.upsample_method == "repeat":
            # Simple repetition: repeat each coarse action N times
            repeat_factor = coarse_frameskip // fine_frameskip
            if repeat_factor <= 0:
                repeat_factor = 1

            upsampled = []
            for t in range(coarse_horizon):
                repeated_action = coarse_actions[:, t:t+1].repeat(1, repeat_factor, 1)
                upsampled.append(repeated_action)

            upsampled_actions = torch.cat(upsampled, dim=1)

            # Truncate or pad to target horizon
            current_len = upsampled_actions.shape[1]
            if current_len > target_horizon:
                upsampled_actions = upsampled_actions[:, :target_horizon]
            elif current_len < target_horizon:
                # Pad with last action
                padding = upsampled_actions[:, -1:].repeat(1, target_horizon - current_len, 1)
                upsampled_actions = torch.cat([upsampled_actions, padding], dim=1)

        elif self.upsample_method == "linear":
            # Linear interpolation between coarse actions
            # First, convert to cumulative time steps
            coarse_times = torch.arange(coarse_horizon) * coarse_frameskip
            fine_times = torch.arange(target_horizon) * fine_frameskip

            # Interpolate for each batch and action dimension
            upsampled_actions = torch.zeros(batch_size, target_horizon, action_dim, 
                                          device=coarse_actions.device)

            for b in range(batch_size):
                for d in range(action_dim):
                    upsampled_actions[b, :, d] = torch.interp(
                        fine_times.float(), 
                        coarse_times.float(), 
                        coarse_actions[b, :, d]
                    )
        else:
            raise ValueError(f"Unknown upsample method: {self.upsample_method}")

        return upsampled_actions

    def _downsample_trajectory(self, trajectory_obs, coarse_frameskip, fine_frameskip):
        """
        Downsample fine trajectory to coarse resolution for tracking.
        
        Args:
            trajectory_obs: dict with fine-resolution observations
            coarse_frameskip: int
            fine_frameskip: int
            
        Returns:
            downsampled_obs: dict with coarse-resolution observations
        """
        downsample_factor = coarse_frameskip // fine_frameskip
        if downsample_factor <= 1:
            return trajectory_obs

        downsampled = {}
        for key, arr in trajectory_obs.items():
            # Take every N-th frame
            downsampled[key] = arr[:, ::downsample_factor]

        return downsampled

    def plan(self, obs_0, obs_g, actions=None):
        """
        Perform hierarchical planning.
        
        Args:
            obs_0: initial observations
            obs_g: goal observations  
            actions: optional initial actions (not used)
            
        Returns:
            actions: (B, T, action_dim) fine-resolution actions
            action_len: (B,) array of action lengths (all inf for now)
        """
        # Step 1: Coarse planning
        print(f"{self.logging_prefix}: Starting coarse planning...")

        # Temporarily modify evaluator frameskip for coarse planning
        original_frameskip = self.evaluator.frameskip
        self.evaluator.frameskip = self.coarse_frameskip

        try:
            coarse_actions, _ = self.coarse_planner.plan(obs_0, obs_g, actions=None)
            print(f"{self.logging_prefix}: Coarse planning completed. Shape: {coarse_actions.shape}")

            # Rollout coarse trajectory to get intermediate states for tracking
            trans_obs_0 = move_to_device(
                self.preprocessor.transform_obs(obs_0), self.device
            )
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            )

            with torch.no_grad():
                coarse_z_obses, _ = self.wm.rollout(
                    obs_0=trans_obs_0,
                    act=coarse_actions,
                )

            # Convert coarse predictions back to observation space for tracking
            coarse_obs_track = {}
            for key in trans_obs_0.keys():
                if key in coarse_z_obses:
                    # Decode visual features back to pixels if needed
                    if key == "visual" and hasattr(self.wm, 'decoder') and self.wm.decoder is not None:
                        coarse_obs_track[key] = self.wm.decoder(coarse_z_obses[key])
                    else:
                        coarse_obs_track[key] = coarse_z_obses[key]

            # Add goal observation as final tracking target
            for key in coarse_obs_track.keys():
                coarse_obs_track[key] = torch.cat([
                    coarse_obs_track[key], 
                    trans_obs_g[key]
                ], dim=1)

        finally:
            # Restore original frameskip
            self.evaluator.frameskip = original_frameskip

        # Step 2: Upsample coarse actions to fine resolution
        fine_horizon = self.fine_planner.horizon
        upsampled_actions = self._upsample_actions(
            coarse_actions.detach(),
            self.coarse_frameskip,
            self.fine_frameskip,
            fine_horizon
        )

        # Step 3: Fine planning with tracking
        print(f"{self.logging_prefix}: Starting fine planning with tracking...")
        print(f"{self.logging_prefix}: Upsampled actions shape: {upsampled_actions.shape}")

        # Create tracking targets by downsampling the coarse trajectory to fine resolution
        tracking_targets = self._downsample_trajectory(
            coarse_obs_track,
            self.coarse_frameskip,
            self.fine_frameskip
        )

        # Override the fine planner's objective to include tracking
        original_objective = self.fine_planner.objective_fn
        self.fine_planner.objective_fn = self._create_tracking_objective_fn()

        try:
            fine_actions, action_len = self.fine_planner.plan(
                obs_0, tracking_targets, actions=upsampled_actions
            )
            print(f"{self.logging_prefix}: Fine planning completed. Shape: {fine_actions.shape}")

        finally:
            # Restore original objective
            self.fine_planner.objective_fn = original_objective

        # Log hierarchical planning metrics
        if self.wandb_run is not None:
            self.wandb_run.log({
                f"{self.logging_prefix}/coarse_horizon": coarse_actions.shape[1],
                f"{self.logging_prefix}/fine_horizon": fine_actions.shape[1],
                f"{self.logging_prefix}/coarse_frameskip": self.coarse_frameskip,
                f"{self.logging_prefix}/fine_frameskip": self.fine_frameskip,
                f"{self.logging_prefix}/tracking_weight": self.tracking_weight,
            })

        return fine_actions, action_len
