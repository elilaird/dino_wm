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
from planning.evaluator import PlanEvaluator
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

def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
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
            print(f"PlanWorkspace: Using discrete actions with {self.action_dim} possible values")

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

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
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
            is_discrete=self.is_discrete,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm, 
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            is_discrete=self.is_discrete,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            # self.planner.n_taken_actions = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict['planner']['n_taken_actions']

        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )

            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None

        elif self.goal_source in ["manual_easy", "manual_medium", "manual_hard"]:
            # set start and goal states manually to farthest points
            TOP_LEFT_STATE = np.array([0.80, 0.80])
            TOP_RIGHT_STATE = np.array([2.80, 0.80])
            BOTTOM_LEFT_STATE = np.array([0.80, 2.75])
            BOTTOM_RIGHT_STATE = np.array([2.80, 2.75])

            if self.goal_source == "manual_easy":
                start_state = TOP_LEFT_STATE
                goal_state = BOTTOM_LEFT_STATE
            elif self.goal_source == "manual_medium":
                start_state = TOP_LEFT_STATE
                goal_state = BOTTOM_RIGHT_STATE
            elif self.goal_source == "manual_hard":
                start_state = TOP_LEFT_STATE
                goal_state = TOP_RIGHT_STATE
            
            # add noise to start and goal states for each of n_evals
            start_states = []
            goal_states = []
            for i in range(self.n_evals):
                start_states.append(np.concatenate([start_state + np.random.normal(0, 0.01, [2]), np.random.uniform(-5.2262554, 5.2262554, [2])]))
                goal_states.append(np.concatenate([goal_state + np.random.normal(0, 0.01, [2]), np.random.uniform(-5.2262554, 5.2262554, [2])]))
            
            start_states = np.stack(start_states)
            goal_states = np.stack(goal_states)

            obs_0, state_0 = self.env.prepare(self.eval_seed, start_states)
            obs_g, state_g = self.env.prepare(self.eval_seed, goal_states)

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = start_state
            self.state_g = goal_state
            self.gt_actions = None

        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
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
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
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

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        # clear gpu memory for wm
        self.wm.zero_grad()

        torch.cuda.empty_cache()

        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )

        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_ckpt_state_dict(snapshot_path, device):
    """Load checkpoint with state dicts and return both state dicts and training config."""
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    
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


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path, weights_only=False)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


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
        models["action_encoder"] = hydra.utils.instantiate(train_cfg.action_encoder, frameskip=train_cfg.frameskip)
        models["action_encoder"].load_state_dict(state_dicts["action_encoder"])
        print(f"Loaded action encoder from state dict checkpoint")
    else:
        print(f"Loaded untrained action encoder from config")
        models["action_encoder"] = hydra.utils.instantiate(train_cfg.action_encoder, frameskip=train_cfg.frameskip)
    
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
                ckpt = torch.load(decoder_path, weights_only=False)
                if isinstance(ckpt, dict):
                    models["decoder"] = ckpt["decoder"]
                else:
                    models["decoder"] = torch.load(decoder_path, weights_only=False)
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


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1]) + f"_{os.environ['SLURM_JOB_ID']}"
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
        # frameskip=model_cfg.frameskip,
        frameskip=cfg_dict["test_frameskip"],
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model_state_dict(model_ckpt, model_cfg, num_action_repeat, device=device)

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs, disable_env_checker=True
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    print("Build workspace")
    print(f"Using frameskip: {cfg_dict['test_frameskip']}")

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=cfg_dict["test_frameskip"],
        # frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    print("Perform planning")
    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    print(cfg_dict)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
