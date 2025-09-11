import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device
from models.encoder.discrete_action_encoder import DiscreteActionEncoder


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        is_discrete=False,
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
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.is_discrete = is_discrete

        if self.is_discrete:
            self.num_actions = self.wm.action_encoder.num_actions
            print(f"CEM: Using discrete actions with {self.num_actions} possible values")

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        
        if self.is_discrete:
            # For discrete actions, mu represents action probabilities
            # Initialize with uniform distribution over actions
            mu = torch.ones([n_evals, self.horizon, self.num_actions]) / self.num_actions
            sigma = torch.ones([n_evals, self.horizon, self.num_actions]) * 0.1
        else:
            # Original continuous action logic
            sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
            if actions is None:
                mu = torch.zeros(n_evals, 0, self.action_dim)
            else:
                mu = actions
            device = mu.device
            t = mu.shape[1]
            remaining_t = self.horizon - t

            if remaining_t > 0:
                new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
                mu = torch.cat([mu, new_mu.to(device)], dim=1)
        
        return mu, sigma

    def sample_discrete_actions(self, mu, sigma, n_samples):
        """Sample discrete actions from categorical distribution"""
        # Convert mu to probabilities using softmax
        probs = torch.softmax(mu, dim=-1)  # (n_evals, horizon, num_actions)
        
        # Sample actions from categorical distribution
        # For each trajectory and timestep, sample n_samples actions
        batch_size, horizon, num_actions = probs.shape
        probs_expanded = probs.unsqueeze(0).expand(n_samples, -1, -1, -1)  # (n_samples, n_evals, horizon, num_actions)
        probs_flat = probs_expanded.reshape(-1, num_actions)  # (n_samples * n_evals * horizon, num_actions)
        
        # Sample from categorical
        actions_flat = torch.multinomial(probs_flat, 1).squeeze(-1)  # (n_samples * n_evals * horizon,)
        actions = actions_flat.reshape(n_samples, batch_size, horizon)  # (n_samples, n_evals, horizon)
        
        # Add action_dim dimension to match expected shape (n_samples, n_evals, horizon, 1)
        actions = actions.unsqueeze(-1)
        
        return actions

    def update_discrete_distribution(self, mu, sigma, topk_actions):
        """Update the categorical distribution parameters"""
        # topk_actions shape: (topk, horizon, 1)
        # Remove the last dimension to get (topk, horizon)
        topk_actions = topk_actions.squeeze(-1)
        
        # Convert topk_actions to one-hot encoding
        batch_size, horizon = mu.shape[:2]
        num_actions = mu.shape[-1]
        
        # Create one-hot encoding of the topk actions
        one_hot = torch.zeros_like(mu)
        for t in range(horizon):
            for action_idx in topk_actions[:, t]:
                one_hot[0, t, action_idx] += 1  # Only one batch element since we're processing one trajectory at a time
        
        # Normalize to get empirical probabilities
        empirical_probs = one_hot / self.topk
        
        # Update mu using exponential moving average
        alpha = 0.1  # Learning rate for distribution update
        new_mu = (1 - alpha) * mu + alpha * torch.log(empirical_probs + 1e-8)
        
        # Update sigma (variance) based on the spread of topk actions
        new_sigma = torch.std(one_hot, dim=-1, keepdim=True).expand_as(mu)
        
        return new_mu, new_sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                
                if self.is_discrete:
                    # Sample discrete actions
                    action = self.sample_discrete_actions(
                        mu[traj:traj+1], sigma[traj:traj+1], self.num_samples
                    ).squeeze(1)  # (num_samples, horizon, 1)
                else:
                    # continuous action sampling
                    action = (
                        torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                            self.device
                        )
                        * sigma[traj]
                        + mu[traj]
                    )
                    action[0] = mu[traj]  # optional: make the first one mu itself
                
                with torch.no_grad():
                    i_z_obses, _ = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                
                if self.is_discrete:
                    # Update discrete distribution
                    mu[traj], sigma[traj] = self.update_discrete_distribution(
                        mu[traj:traj+1], sigma[traj:traj+1], topk_action
                    )
                    mu[traj] = mu[traj].squeeze(0)
                    sigma[traj] = sigma[traj].squeeze(0)
                else:
                    # Original continuous update
                    mu[traj] = topk_action.mean(dim=0)
                    sigma[traj] = topk_action.std(dim=0)

                # cleanup
                del cur_trans_obs_0, cur_z_obs_g, action, i_z_obses
                torch.cuda.empty_cache()  # Force CUDA memory cleanup

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                # Convert mu to discrete actions for evaluation
                if self.is_discrete:
                    eval_actions = torch.argmax(mu, dim=-1).unsqueeze(-1)  # (n_evals, horizon, 1)
                else:
                    eval_actions = mu

                logs, successes, _, _ = self.evaluator.eval_actions(
                    eval_actions, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        # Return final actions
        if self.is_discrete:
            final_actions = torch.argmax(mu, dim=-1).unsqueeze(-1)  # (n_evals, horizon, 1)
        else:
            final_actions = mu
            
        return final_actions, np.full(n_evals, np.inf)  # all actions are valid
