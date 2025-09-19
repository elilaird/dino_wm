import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


def one_hot(indices: torch.Tensor, K: int):
    """Create one-hot encoding from indices"""
    y = torch.zeros(*indices.shape, K, device=indices.device, dtype=indices.dtype)
    return y.scatter_(-1, indices.unsqueeze(-1), 1)


def straight_through(p_soft):
    """Straight-through estimator: forward uses hard one-hot, backward uses soft gradient"""
    idx = p_soft.argmax(dim=-1)              # (B, T)
    p_hard = one_hot(idx, p_soft.size(-1))   # (B, T, K)
    return p_hard + (p_soft - p_soft.detach())


class GDDiscretePlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        lr,
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
        K=None,  # number of discrete actions
        use_st_forward=True,
        tau_start=1.5,
        tau_end=0.3,
        lam_entropy=0.0,
        lam_onehot=0.0,
        lam_switch=0.0,
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
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.use_st_forward = use_st_forward
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.lam_entropy = lam_entropy
        self.lam_onehot = lam_onehot
        self.lam_switch = lam_switch
        self.K = K  # must be set for discrete

        # Get action embedding from WM if available
        self.action_embed = getattr(self.wm, "action_embed", None)
        if self.action_embed is None:
            # Create identity embedding (one-hot as features)
            self.action_embed = torch.nn.Embedding(self.K, self.K)
            self.action_embed.weight.data = torch.eye(self.K)
            self.action_embed.weight.requires_grad = False

    def init_logits(self, obs_0, logits=None):
        """Initialize or pad logits for planning"""
        n_evals = obs_0["visual"].shape[0]
        if logits is None:
            # Initialize with zeros (uniform distribution)
            logits = torch.zeros(n_evals, self.horizon, self.K, device=self.device)
        else:
            logits = logits.to(self.device)
            t = logits.shape[1]
            if t < self.horizon:
                # Pad with zeros
                pad = torch.zeros(logits.size(0), self.horizon - t, self.K, device=logits.device)
                logits = torch.cat([logits, pad], dim=1)
            elif t > self.horizon:
                # Truncate to horizon
                logits = logits[:, :self.horizon]
        logits.requires_grad_(True)
        return logits

    def get_action_optimizer(self, logits):
        return torch.optim.Adam([logits], lr=self.lr)

    def _probs_from_logits(self, L, it):
        """Convert logits to probabilities with temperature annealing"""
        tau = self.tau_start + (self.tau_end - self.tau_start) * (it / max(1, self.opt_steps - 1))
        p = torch.softmax(L / tau, dim=-1)
        return p, tau

    def _embed_actions(self, probs, straight_through=False):
        """Convert probabilities to action embeddings"""
        if straight_through:
            p_st = straight_through(probs)     # (B, T, K)
            a = p_st @ self.action_embed.weight  # (B, T, d)
        else:
            a = probs @ self.action_embed.weight  # (B, T, d)
        return a

    def _regularizers(self, p):
        """Compute regularization terms"""
        # Entropy regularization (encourage sharper distributions)
        ent = -(p.clamp_min(1e-8) * (p.clamp_min(1e-8)).log()).sum(-1).mean()
        
        # One-hot penalty (encourage discrete actions)
        onehot_pen = 1.0 - (p * p).sum(-1).mean()
        
        # Switch penalty (reduce action chattering)
        switch_pen = (p[:, 1:] - p[:, :-1]).abs().mean()
        
        return ent, onehot_pen, switch_pen

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            obs_0: initial observations
            obs_g: goal observations
            actions: optional warm-start logits
        Returns:
            actions: (B, T, K) one-hot actions for execution
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        # Initialize logits (treating 'actions' as warm-start logits if provided)
        logits = self.init_logits(obs_0, logits=actions)
        optimizer = self.get_action_optimizer(logits)
        n_evals = logits.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()

            # Convert logits to probabilities
            p, tau = self._probs_from_logits(logits, i)
            
            # Choose forward type: soft embedding or straight-through embedding
            a_plan = self._embed_actions(p, straight_through=self.use_st_forward)

            # Rollout latent dynamics with planned (relaxed) actions
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=a_plan,
            )
            
            # Compute planning loss
            loss_vec = self.objective_fn(i_z_obses, z_obs_g_detached)  # (B,)
            cost = loss_vec.mean() * n_evals

            # Optional regularizers
            ent, onehot_pen, switch_pen = self._regularizers(p)
            total_loss = (cost
                          + self.lam_entropy * (-ent)   # subtract entropy => sharper p
                          + self.lam_onehot * onehot_pen
                          + self.lam_switch * switch_pen)

            total_loss.backward()
            
            # Monitor gradients for vanishing/exploding gradients
            if logits.grad is not None:
                grad_norm = logits.grad.norm()
                grad_max = logits.grad.abs().max()
                grad_mean = logits.grad.abs().mean()
                grad_std = logits.grad.std()
                
                # Check for vanishing gradients
                vanishing_threshold = 1e-8
                is_vanishing = grad_norm < vanishing_threshold
                
                # Check for exploding gradients
                exploding_threshold = 1e3
                is_exploding = grad_norm > exploding_threshold
                
                # Log gradient statistics
                grad_logs = {
                    f"{self.logging_prefix}/grad_norm": grad_norm.item(),
                    f"{self.logging_prefix}/grad_max": grad_max.item(),
                    f"{self.logging_prefix}/grad_mean": grad_mean.item(),
                    f"{self.logging_prefix}/grad_std": grad_std.item(),
                    f"{self.logging_prefix}/grad_vanishing": is_vanishing,
                    f"{self.logging_prefix}/grad_exploding": is_exploding,
                }
                self.wandb_run.log(grad_logs)
            
            optimizer.step()

            # Logging
            self.wandb_run.log({
                f"{self.logging_prefix}/loss": total_loss.item(),
                f"{self.logging_prefix}/tau": tau,
                f"{self.logging_prefix}/entropy": ent.item(),
                f"{self.logging_prefix}/onehot_pen": onehot_pen.item(),
                f"{self.logging_prefix}/switch_pen": switch_pen.item(),
                "step": i + 1
            })

            # Evaluation
            if self.evaluator is not None and i % self.eval_every == 0:
                # For evaluator/env, commit HARD actions (one-hot) for the current plan
                hard_idx = p.argmax(dim=-1)  # (B, T)
                hard_oh = one_hot(hard_idx, self.K).float()
                
                logs, successes, _, _ = self.evaluator.eval_actions(
                    hard_oh.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        # Return final hard actions for execution
        with torch.no_grad():
            p_final, _ = self._probs_from_logits(logits, self.opt_steps - 1)
            hard_idx = p_final.argmax(dim=-1)   # (B, T)
            hard_oh = one_hot(hard_idx, self.K).float()
        
        return hard_oh, np.full(n_evals, np.inf)  # all actions are valid
