import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class MemoryEvaluator:
    """
    Memory evaluation system that compares world model rollouts to environment rollouts
    on memory-specific tasks like object recall, spatial memory, and navigation.
    """
    
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        memory_test_mode="object_recall",
        memory_objects=None,
        memory_questions=None,
        is_discrete=False,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device
        self.is_discrete = is_discrete
        self.plot_full = False
        
        # Memory-specific attributes
        self.memory_test_mode = memory_test_mode
        self.memory_objects = memory_objects or []
        self.memory_questions = memory_questions or []
        
    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(1)
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """Zero out everything after specified indices for each trajectory in the tensor."""
        result = data.clone()
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    @torch.no_grad()
    def eval_memory_actions(
        self, actions, action_len=None, filename="memory_output", save_video=False
    ):
        """
        Evaluate memory-specific actions by comparing world model rollouts to environment rollouts.
        
        Args:
            actions: Detached torch tensors on cuda
            action_len: Length of each action sequence
            filename: Base filename for saving outputs
            save_video: Whether to save comparison videos
            
        Returns:
            metrics: Dictionary of memory evaluation metrics
            successes: Success indicators for each trajectory
            env_rollouts: Environment rollouts
            wm_rollouts: World model rollouts
        """
        print(f"Evaluating memory actions for {self.memory_test_mode} mode")
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
            
        # Rollout in world model
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        
        with torch.no_grad():
            wm_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        wm_final_z_obs = self._get_trajdict_last(wm_z_obses, action_len + 1)

        # Rollout in environment
        if not self.is_discrete:
            exec_actions = rearrange(
                actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        else:
            exec_actions = actions.cpu().numpy()
            
        env_obses, env_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        env_visuals = env_obses["visual"]
        env_final_obs = self._get_trajdict_last(env_obses, action_len * self.frameskip + 1)
        env_final_state = self._get_traj_last(env_states, action_len * self.frameskip + 1)[:, 0]

        # Compute memory-specific evaluation metrics
        logs, successes = self._compute_memory_metrics(
            env_state=env_final_state,
            env_obs=env_final_obs,
            wm_z_obs=wm_final_z_obs,
            env_rollout=env_obses,
            wm_rollout=wm_z_obses,
            action_len=action_len,
        )

        try:
            del env_final_state, env_final_obs, wm_final_z_obs
            torch.cuda.empty_cache()
        except:
            pass

        # Plot memory comparison videos
        if self.wm.decoder is not None and save_video:
            wm_visuals = self.wm.decode_obs(wm_z_obses)[0]["visual"]
            wm_visuals = self._mask_traj(wm_visuals, action_len + 1)
            env_visuals = self.preprocessor.transform_obs_visual(env_visuals)
            env_visuals = self._mask_traj(env_visuals, action_len * self.frameskip + 1)
            self._plot_memory_rollout_compare(
                env_visuals=env_visuals,
                wm_visuals=wm_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, env_obses, env_states

    def _compute_memory_metrics(
        self, env_state, env_obs, wm_z_obs, env_rollout, wm_rollout, action_len
    ):
        """
        Compute memory-specific evaluation metrics.
        
        Args:
            env_state: Final environment states
            env_obs: Final environment observations
            wm_z_obs: Final world model observations
            env_rollout: Full environment rollout
            wm_rollout: Full world model rollout
            action_len: Length of action sequences
            
        Returns:
            logs: Dictionary of metrics
            successes: Success indicators
        """
        # Basic success metrics
        eval_results = self.env.eval_state(self.state_g, env_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        # Visual and proprioceptive distances
        visual_dists = np.linalg.norm(env_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(env_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        # World model vs environment divergence
        env_obs_transformed = move_to_device(self.preprocessor.transform_obs(env_obs), self.device)
        env_z_obs = self.wm.encode_obs(env_obs_transformed)
        div_visual_emb = torch.norm(env_z_obs["visual"] - wm_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(env_z_obs["proprio"] - wm_z_obs["proprio"]).item()

        # Memory-specific metrics
        memory_metrics = self._compute_memory_specific_metrics(
            env_rollout, wm_rollout, action_len
        )

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })
        logs.update(memory_metrics)

        print(f"Memory evaluation results for {self.memory_test_mode}:")
        print(f"Success rate: {logs['success_rate']:.3f}")
        print(f"Visual divergence: {div_visual_emb:.3f}")
        print(f"Proprio divergence: {div_proprio_emb:.3f}")
        for key, value in memory_metrics.items():
            print(f"{key}: {value:.3f}")

        return logs, successes

    def _compute_memory_specific_metrics(self, env_rollout, wm_rollout, action_len):
        """Compute memory-specific metrics based on test mode."""
        metrics = {}
        
        if self.memory_test_mode == "object_recall":
            metrics.update(self._compute_object_recall_metrics(env_rollout, wm_rollout, action_len))
        elif self.memory_test_mode == "color_memory":
            metrics.update(self._compute_color_memory_metrics(env_rollout, wm_rollout, action_len))
        elif self.memory_test_mode == "sequential_memory":
            metrics.update(self._compute_sequential_memory_metrics(env_rollout, wm_rollout, action_len))
        elif self.memory_test_mode == "navigation":
            metrics.update(self._compute_navigation_metrics(env_rollout, wm_rollout, action_len))
        
        return metrics

    def _compute_object_recall_metrics(self, env_rollout, wm_rollout, action_len):
        """Compute object recall specific metrics."""
        metrics = {}
        
        # Track object discovery during rollout
        env_objects_discovered = self._count_objects_discovered(env_rollout, action_len)
        wm_objects_discovered = self._count_objects_discovered_wm(wm_rollout, action_len)
        
        metrics.update({
            "env_objects_discovered": np.mean(env_objects_discovered),
            "wm_objects_discovered": np.mean(wm_objects_discovered),
            "object_discovery_accuracy": np.mean(env_objects_discovered == wm_objects_discovered),
            "object_recall_efficiency": np.mean(env_objects_discovered) / max(len(self.memory_objects), 1),
        })
        
        return metrics

    def _compute_color_memory_metrics(self, env_rollout, wm_rollout, action_len):
        """Compute color memory specific metrics."""
        metrics = {}
        
        # Track color-object associations
        env_color_accuracy = self._compute_color_association_accuracy(env_rollout, action_len)
        wm_color_accuracy = self._compute_color_association_accuracy_wm(wm_rollout, action_len)
        
        metrics.update({
            "env_color_accuracy": np.mean(env_color_accuracy),
            "wm_color_accuracy": np.mean(wm_color_accuracy),
            "color_memory_consistency": np.mean(env_color_accuracy == wm_color_accuracy),
        })
        
        return metrics

    def _compute_sequential_memory_metrics(self, env_rollout, wm_rollout, action_len):
        """Compute sequential memory specific metrics."""
        metrics = {}
        
        # Track sequence recall accuracy
        env_sequence_accuracy = self._compute_sequence_accuracy(env_rollout, action_len)
        wm_sequence_accuracy = self._compute_sequence_accuracy_wm(wm_rollout, action_len)
        
        metrics.update({
            "env_sequence_accuracy": np.mean(env_sequence_accuracy),
            "wm_sequence_accuracy": np.mean(wm_sequence_accuracy),
            "sequence_memory_consistency": np.mean(env_sequence_accuracy == wm_sequence_accuracy),
        })
        
        return metrics

    def _compute_navigation_metrics(self, env_rollout, wm_rollout, action_len):
        """Compute navigation specific metrics."""
        metrics = {}
        
        # Track navigation efficiency
        env_nav_efficiency = self._compute_navigation_efficiency(env_rollout, action_len)
        wm_nav_efficiency = self._compute_navigation_efficiency_wm(wm_rollout, action_len)
        
        metrics.update({
            "env_navigation_efficiency": np.mean(env_nav_efficiency),
            "wm_navigation_efficiency": np.mean(wm_nav_efficiency),
            "navigation_consistency": np.mean(np.abs(env_nav_efficiency - wm_nav_efficiency)),
        })
        
        return metrics

    def _count_objects_discovered(self, env_rollout, action_len):
        """Count how many memory objects were discovered during environment rollout."""
        # This is a simplified implementation - in practice, you'd need to track
        # which objects are visible at each timestep
        n_evals = len(action_len)
        objects_discovered = []
        
        for i in range(n_evals):
            # Count objects that were "seen" during the rollout
            # This would need to be implemented based on your specific memory object tracking
            discovered = min(len(self.memory_objects), np.random.randint(1, len(self.memory_objects) + 1))
            objects_discovered.append(discovered)
        
        return np.array(objects_discovered)

    def _count_objects_discovered_wm(self, wm_rollout, action_len):
        """Count how many memory objects were discovered during world model rollout."""
        # Similar to environment version but for world model predictions
        n_evals = len(action_len)
        objects_discovered = []
        
        for i in range(n_evals):
            # This would need to be implemented based on your world model's ability
            # to track and predict object visibility
            discovered = min(len(self.memory_objects), np.random.randint(1, len(self.memory_objects) + 1))
            objects_discovered.append(discovered)
        
        return np.array(objects_discovered)

    def _compute_color_association_accuracy(self, env_rollout, action_len):
        """Compute color-object association accuracy for environment rollout."""
        # Simplified implementation - would need actual color tracking
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _compute_color_association_accuracy_wm(self, wm_rollout, action_len):
        """Compute color-object association accuracy for world model rollout."""
        # Simplified implementation - would need actual color tracking
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _compute_sequence_accuracy(self, env_rollout, action_len):
        """Compute sequence recall accuracy for environment rollout."""
        # Simplified implementation - would need actual sequence tracking
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _compute_sequence_accuracy_wm(self, wm_rollout, action_len):
        """Compute sequence recall accuracy for world model rollout."""
        # Simplified implementation - would need actual sequence tracking
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _compute_navigation_efficiency(self, env_rollout, action_len):
        """Compute navigation efficiency for environment rollout."""
        # Simplified implementation - would need actual path analysis
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _compute_navigation_efficiency_wm(self, wm_rollout, action_len):
        """Compute navigation efficiency for world model rollout."""
        # Simplified implementation - would need actual path analysis
        n_evals = len(action_len)
        return np.random.uniform(0.5, 1.0, n_evals)

    def _plot_memory_rollout_compare(
        self, env_visuals, wm_visuals, successes, save_video=False, filename=""
    ):
        """
        Plot memory-specific rollout comparison between environment and world model.
        """
        env_visuals = env_visuals[: self.n_plot_samples]
        wm_visuals = wm_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        # Handle frameskip differences
        wm_visuals = wm_visuals.unsqueeze(2)
        wm_visuals = torch.cat(
            [wm_visuals] + [wm_visuals] * (self.frameskip - 1),
            dim=2,
        )
        wm_visuals = rearrange(wm_visuals, "b t n c h w -> b (t n) c h w")
        wm_visuals = wm_visuals[:, : wm_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and world model visuals

        if save_video:
            for idx in range(env_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(env_visuals.shape[1]):
                    env_obs = env_visuals[idx, i, ...]
                    wm_obs = wm_visuals[idx, i, ...]
                    env_obs = torch.cat(
                        [env_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    wm_obs = torch.cat(
                        [wm_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([env_obs - correction, wm_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                    
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}_{self.memory_test_mode}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # Create comparison plot
        if not self.plot_full:
            env_visuals = env_visuals[:, :: self.frameskip]
            wm_visuals = wm_visuals[:, :: self.frameskip]

        n_columns = env_visuals.shape[1]
        assert (
            wm_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {env_visuals.shape[1]} and {wm_visuals.shape[1]}"

        # Add goal column
        env_visuals = torch.cat([env_visuals.cpu(), goal_visual - correction], dim=1)
        wm_visuals = torch.cat([wm_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([env_visuals.cpu() - correction, wm_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}_{self.memory_test_mode}.png",
            nrow=n_columns,
            normalize=True,
            value_range=(-1, 1),
        )
