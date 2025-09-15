import os
import gymnasium as gym
import json
import hydra
import random
import torch
import pickle
from models.encoder.discrete_action_encoder import DiscreteActionEncoder
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.memory_evaluator import MemoryEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def build_memory_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    memory_test_mode=["object_recall", "color_memory", "sequential_memory"],
    n_memory_objects=[3, 4, 5],
    memory_object_types=[["ball", "box", "key"]],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of memory plan overrides for different memory test configurations.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "memory_test_mode": mode,
            "n_memory_objects": n_objects,
            "memory_object_types": obj_types,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for mode, n_objects, obj_types, g_H, a in product(
            memory_test_mode, n_memory_objects, memory_object_types, goal_H, alpha
        )
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class MemoryPlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # Memory-specific configuration
        self.memory_test_mode = cfg_dict["memory_test_mode"]
        self.n_memory_objects = cfg_dict["n_memory_objects"]
        self.memory_object_types = cfg_dict["memory_object_types"]

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        self.is_discrete = isinstance(self.wm.action_encoder, DiscreteActionEncoder)
        if self.is_discrete:
            self.action_dim = self.wm.action_encoder.num_actions
            print(f"MemoryPlanWorkspace: Using discrete actions with {self.action_dim} possible values")

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        # Prepare memory-specific targets
        self.prepare_memory_targets()

        self.evaluator = MemoryEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            memory_test_mode=self.memory_test_mode,
            memory_objects=self.memory_objects,
            memory_questions=self.memory_questions,
            is_discrete=self.is_discrete,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "memory_logs.json"
        
        # For memory evaluation, we use scripted trajectories instead of planning
        self.scripted_trajectories = self._generate_scripted_trajectories()

        self.dump_memory_targets()

    def prepare_memory_targets(self):
        """Prepare memory-specific targets and trajectories."""
        states = []
        actions = []
        observations = []

        # Update env config from val trajs
        observations, states, actions, env_info = (
            self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
        )
        self.env.update_env(env_info)

        # Configure environment for memory testing
        if hasattr(self.env, 'set_memory_test_mode'):
            self.env.set_memory_test_mode(self.memory_test_mode)
        if hasattr(self.env, 'set_n_memory_objects'):
            self.env.set_n_memory_objects(self.n_memory_objects)
        if hasattr(self.env, 'set_memory_object_types'):
            self.env.set_memory_object_types(self.memory_object_types)

        # Get states from val trajs
        init_state = [x[0] for x in states]
        init_state = np.array(init_state)
        actions = torch.stack(actions)
        
        # For memory testing, we might want to use specific action sequences
        if self.goal_source == "memory_exploration":
            actions = self._generate_memory_exploration_actions(actions)
        elif self.goal_source == "random_action":
            actions = torch.randn_like(actions)
            
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
        exec_actions = self.data_preprocessor.denormalize_actions(actions)
        
        # Replay actions in env to get ground truth observations
        rollout_obses, rollout_states = self.env.rollout(
            self.eval_seed, init_state, exec_actions.numpy()
        )
        
        self.obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.state_0 = init_state
        self.state_g = rollout_states[:, -1]
        self.gt_actions = wm_actions

        # Extract memory objects and questions from environment
        if hasattr(self.env, 'get_memory_objects'):
            self.memory_objects = self.env.get_memory_objects()
        else:
            self.memory_objects = []
            
        if hasattr(self.env, 'get_memory_questions'):
            self.memory_questions = self.env.get_memory_questions()
        else:
            self.memory_questions = []

    def _generate_memory_exploration_actions(self, base_actions):
        """Generate actions specifically designed for memory exploration."""
        # This would generate actions that systematically explore the environment
        # to discover memory objects
        return base_actions  # Placeholder

    def _generate_scripted_trajectories(self):
        """Generate scripted trajectories for memory testing."""
        scripted_trajectories = []
        
        # Generate different types of scripted trajectories based on memory test mode
        if self.memory_test_mode == "object_recall":
            scripted_trajectories = self._generate_object_recall_trajectories()
        elif self.memory_test_mode == "color_memory":
            scripted_trajectories = self._generate_color_memory_trajectories()
        elif self.memory_test_mode == "sequential_memory":
            scripted_trajectories = self._generate_sequential_memory_trajectories()
        elif self.memory_test_mode == "navigation":
            scripted_trajectories = self._generate_navigation_trajectories()
        
        return scripted_trajectories

    def _generate_object_recall_trajectories(self):
        """Generate trajectories for object recall testing."""
        trajectories = []
        for i in range(self.n_evals):
            # Generate exploration trajectory that visits different rooms
            traj = self._generate_exploration_trajectory()
            trajectories.append(traj)
        return trajectories

    def _generate_color_memory_trajectories(self):
        """Generate trajectories for color memory testing."""
        trajectories = []
        for i in range(self.n_evals):
            # Generate trajectory that focuses on color-object associations
            traj = self._generate_color_focused_trajectory()
            trajectories.append(traj)
        return trajectories

    def _generate_sequential_memory_trajectories(self):
        """Generate trajectories for sequential memory testing."""
        trajectories = []
        for i in range(self.n_evals):
            # Generate trajectory that tests sequence memory
            traj = self._generate_sequence_trajectory()
            trajectories.append(traj)
        return trajectories

    def _generate_navigation_trajectories(self):
        """Generate trajectories for navigation testing."""
        trajectories = []
        for i in range(self.n_evals):
            # Generate navigation trajectory
            traj = self._generate_navigation_trajectory()
            trajectories.append(traj)
        return trajectories

    def _generate_exploration_trajectory(self):
        """Generate a systematic exploration trajectory."""
        # This would generate actions that systematically explore all rooms
        # to discover memory objects
        return torch.randn(self.goal_H, self.action_dim)

    def _generate_color_focused_trajectory(self):
        """Generate a trajectory focused on color-object associations."""
        # This would generate actions that focus on observing colored objects
        return torch.randn(self.goal_H, self.action_dim)

    def _generate_sequence_trajectory(self):
        """Generate a trajectory that tests sequence memory."""
        # This would generate actions that test temporal sequence memory
        return torch.randn(self.goal_H, self.action_dim)

    def _generate_navigation_trajectory(self):
        """Generate a navigation trajectory."""
        # This would generate actions for navigation tasks
        return torch.randn(self.goal_H, self.action_dim)

    def sample_traj_segment_from_dset(self, traj_len):
        """Sample trajectory segments from dataset."""
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def dump_memory_targets(self):
        """Dump memory-specific targets for debugging."""
        with open("memory_plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                    "memory_test_mode": self.memory_test_mode,
                    "memory_objects": self.memory_objects,
                    "memory_questions": self.memory_questions,
                },
                f,
            )
        file_path = os.path.abspath("memory_plan_targets.pkl")
        print(f"Dumped memory plan targets to {file_path}")

    def perform_memory_evaluation(self):
        """Perform memory evaluation using scripted trajectories."""
        print(f"Performing memory evaluation for {self.memory_test_mode} mode")
        
        all_logs = []
        all_successes = []
        
        for traj_idx, trajectory in enumerate(self.scripted_trajectories):
            print(f"Evaluating trajectory {traj_idx + 1}/{len(self.scripted_trajectories)}")
            
            # Convert trajectory to proper format
            actions = trajectory.unsqueeze(0).to(self.device)  # Add batch dimension
            action_len = torch.tensor([len(trajectory)], device=self.device)
            
            # Evaluate this trajectory
            logs, successes, env_rollouts, wm_rollouts = self.evaluator.eval_memory_actions(
                actions, action_len, 
                filename=f"memory_output_{self.memory_test_mode}_{traj_idx}",
                save_video=True
            )
            
            all_logs.append(logs)
            all_successes.extend(successes)
        
        # Aggregate results across all trajectories
        aggregated_logs = self._aggregate_memory_logs(all_logs)
        aggregated_logs = {f"memory_eval/{k}": v for k, v in aggregated_logs.items()}
        
        # Log to wandb
        self.wandb_run.log(aggregated_logs)
        
        # Save to file
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in aggregated_logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        
        return aggregated_logs

    def _aggregate_memory_logs(self, all_logs):
        """Aggregate logs from multiple trajectory evaluations."""
        if not all_logs:
            return {}
        
        aggregated = {}
        for key in all_logs[0].keys():
            values = [logs[key] for logs in all_logs if key in logs]
            if values:
                if isinstance(values[0], (int, float, np.number)):
                    aggregated[key] = np.mean(values)
                else:
                    aggregated[key] = values
        
        return aggregated


def load_ckpt_state_dict(snapshot_path, device):
    """Load checkpoint with state dicts and return both state dicts and training config."""
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    
    loaded_keys = []
    state_dicts = {}
    train_cfg = None
    
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            state_dicts[k] = v
        elif k == "train_cfg":
            train_cfg = v
        elif k == "epoch":
            pass  # epoch is handled separately
    
    result = {
        "state_dicts": state_dicts,
        "train_cfg": train_cfg,
        "epoch": payload.get("epoch", 0),
        "loaded_keys": loaded_keys
    }
    return result


def load_model_state_dict(model_ckpt, train_cfg, num_action_repeat, device):
    """Load model using state dicts for safer DDP loading."""
    # First try to load with state dicts
    if model_ckpt.exists():
        try:
            ckpt_data = load_ckpt_state_dict(model_ckpt, device)
            state_dicts = ckpt_data["state_dicts"]
            epoch = ckpt_data["epoch"]
            print(f"Loading from state dict checkpoint epoch {epoch}: {model_ckpt}")
            
            # Use training config from checkpoint if available, otherwise use provided config
            if ckpt_data["train_cfg"] is not None:
                train_cfg = ckpt_data["train_cfg"]
                print("Using training config from checkpoint")
        except Exception as e:
            print(f"Failed to load state dict checkpoint: {e}")
            print("Falling back to legacy checkpoint loading")
            return load_model(model_ckpt, train_cfg, num_action_repeat, device)
    else:
        state_dicts = {}
        epoch = 0

    # Instantiate models using hydra config
    models = {}
    
    if "encoder" in state_dicts:
        models["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
        models["encoder"].load_state_dict(state_dicts["encoder"])
        print(f"Loaded encoder from state dict checkpoint")
    else:
        models["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
        print(f"Loaded untrained encoder from config")
    
    if "predictor" in state_dicts:
        models["predictor"] = hydra.utils.instantiate(train_cfg.predictor)
        models["predictor"].load_state_dict(state_dicts["predictor"])
        print(f"Loaded predictor from state dict checkpoint")
    else:
        if not hasattr(train_cfg, 'predictor'):
            raise ValueError("Predictor config not found in training config")
        models["predictor"] = hydra.utils.instantiate(train_cfg.predictor)
        print(f"Loaded untrained predictor from config")
    
    if "proprio_encoder" in state_dicts:
        models["proprio_encoder"] = hydra.utils.instantiate(train_cfg.proprio_encoder)
        models["proprio_encoder"].load_state_dict(state_dicts["proprio_encoder"])
        print(f"Loaded proprio encoder from state dict checkpoint")
    else:
        print(f"Loaded untrained proprio encoder from config")
        models["proprio_encoder"] = hydra.utils.instantiate(train_cfg.proprio_encoder)
    
    if "action_encoder" in state_dicts:
        models["action_encoder"] = hydra.utils.instantiate(train_cfg.action_encoder)
        models["action_encoder"].load_state_dict(state_dicts["action_encoder"])
        print(f"Loaded action encoder from state dict checkpoint")
    else:
        print(f"Loaded untrained action encoder from config")
        models["action_encoder"] = hydra.utils.instantiate(train_cfg.action_encoder)
    
    # Handle decoder
    if train_cfg.has_decoder:
        if "decoder" in state_dicts:
            models["decoder"] = hydra.utils.instantiate(train_cfg.decoder)
            models["decoder"].load_state_dict(state_dicts["decoder"])
            print(f"Loaded decoder from state dict checkpoint")
        else:
            # Try to load from separate decoder path
            base_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Loaded untrained decoder from config")
            if train_cfg.env.decoder_path is not None:
                decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
                ckpt = torch.load(decoder_path)
                if isinstance(ckpt, dict):
                    models["decoder"] = ckpt["decoder"]
                else:
                    models["decoder"] = torch.load(decoder_path)
            else:
                raise ValueError(
                    "Decoder path not found in model checkpoint and is not provided in config"
                )
    else:
        models["decoder"] = None

    # Instantiate the full model
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=models["encoder"],
        proprio_encoder=models["proprio_encoder"],
        action_encoder=models["action_encoder"],
        predictor=models["predictor"],
        decoder=models["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def memory_planning_main(cfg_dict):
    """Main function for memory planning evaluation."""
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"memory_eval_{cfg_dict['memory_test_mode']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("memory_outputs/")[-1]) + f"_{os.environ.get('SLURM_JOB_ID', 'local')}"
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model_state_dict(model_ckpt, model_cfg, num_action_repeat, device=device)

    # Create environment with memory testing capabilities
    if model_cfg.env.name == "four_rooms":
        # Use the memory-enabled four rooms environment
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    "four_rooms_memory", 
                    memory_test_mode=cfg_dict["memory_test_mode"],
                    n_memory_objects=cfg_dict["n_memory_objects"],
                    memory_object_types=cfg_dict["memory_object_types"],
                    *model_cfg.env.args, 
                    **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        # Fallback to regular environment
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    print("Build memory plan workspace")
    memory_plan_workspace = MemoryPlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    print("Perform memory evaluation")
    logs = memory_plan_workspace.perform_memory_evaluation()
    return logs


@hydra.main(config_path="conf", config_name="memory_plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Memory evaluation result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    print(cfg_dict)
    cfg_dict["wandb_logging"] = True
    memory_planning_main(cfg_dict)


if __name__ == "__main__":
    main()
