import os
import time
import hydra
from numpy.random import logseries
import psutil
import torch
import wandb
import logging
import warnings
import itertools
import numpy as np
from tqdm import tqdm
import socket
import csv


import multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torchvision import utils
from pathlib import Path
from collections import OrderedDict, defaultdict
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from schedulers import CosineAnnealingWarmRestartsDecay

CTX = mp.get_context("spawn")

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def get_all_process_memory_mb():
    """Return dict with current process memory and total across all children (in MB)."""
    proc = psutil.Process(os.getpid())
    mems = {}

    # Main process
    mems["main_process_gb"] = proc.memory_info().rss / 1024**3

    # Child processes (e.g., dataloader workers, torchrun processes)
    child_memories = [
        p.memory_info().rss for p in proc.children(recursive=True)
    ]
    mems["child_processes_gb"] = sum(child_memories) / 1024**3
    mems["total_gb"] = mems["main_process_gb"] + mems["child_processes_gb"]
    return mems


def get_memory_str():
    """Get current GPU memory usage (PyTorch view) as a string"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return (
            f"GPU: {allocated:.2f} GB alloc "
            f"(reserved: {reserved:.2f} GB, max: {max_allocated:.2f} GB)"
        )
    return "GPU: NA"


def _tensor_nbytes(x):
    # Works for torch.Tensor or numpy.ndarray
    try:
        if isinstance(x, torch.Tensor):
            # numel * bytes per element; for non-contiguous views this is still fine for logical size
            return x.numel() * x.element_size()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.nbytes
    # Fallback: try to infer from shape & dtype
    if (
        hasattr(x, "shape")
        and hasattr(x, "dtype")
        and hasattr(x.dtype, "itemsize")
    ):
        import numpy as np

        return int(np.prod(x.shape)) * int(x.dtype.itemsize)
    return 0


def estimate_batch_memory_str(obs, act=None, state=None):
    """Estimate total bytes of the provided batch tensors (inputs only)."""
    total_bytes = 0

    # obs can be a tensor, array, or a dict of them
    if obs is not None:
        if isinstance(obs, dict):
            for v in obs.values():
                total_bytes += _tensor_nbytes(v)
        else:
            total_bytes += _tensor_nbytes(obs)

    total_bytes += _tensor_nbytes(act) if act is not None else 0
    total_bytes += _tensor_nbytes(state) if state is not None else 0

    gb = total_bytes / (1024**3)
    return f"Batch: {gb:.2f} GB"


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_size = self.cfg.num_hist + self.cfg.num_pred
        self.overlap_size = getattr(self.cfg, "overlap_size", 0)
        self.step_size = self.cfg.step_size
        print(f"Overlap size: {self.overlap_size}")
        print(f"Step size: {self.step_size}")
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if self.cfg.model.train_encoder:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True 
            )
        else:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=False
            )

        self.accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[ddp_kwargs]
        )

        global_rank = self.accelerator.process_index  # global rank
        local_rank = (
            self.accelerator.local_process_index
        )  # rank within the node
        world_size = self.accelerator.num_processes  # total processes
        num_machines = int(
            os.environ.get(
                "ACCELERATE_NUM_MACHINES", os.environ.get("SLURM_NNODES", "1")
            )
        )
        procs_per_machine = max(1, world_size // num_machines)
        machine_rank = global_rank // procs_per_machine
        self.accelerator.print(
            f"world_size={world_size} "
            f"host={socket.gethostname()} "
            f"machine_rank={machine_rank} "
            f"local_rank={local_rank} "
            f"global_rank={global_rank} "
            f"is_main={self.accelerator.is_main_process}"
        )

        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")

        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = (
            self.cfg.training.num_reconstruct_samples
        )
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        # OmegaConf.set_struct(cfg, False)
        with open_dict(cfg):
            if cfg.dry_run:
                cfg.training.batch_size = 32
            cfg.effective_batch_size = cfg.training.batch_size
            cfg.gpu_batch_size = (
                cfg.training.batch_size // self.accelerator.num_processes
            )
        # OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:

            # Try to get SLURM job ID, fallback to readable datetime
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            wandb_run_id = f"dino_wm_{slurm_job_id}"
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            run_name = "{}".format(model_name) + f"_{slurm_job_id}"
            if self.cfg.dry_run:
                run_name += "_dry_run"
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                    name=run_name,
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                    name=run_name,
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
            num_frames=self.cfg.num_frames,
            include_test=True,
        )

        if self.accelerator.is_main_process:
            # print length of train and valid datasets
            log.info(f"Train dataset length: {len(self.datasets['train'])}")
            log.info(f"Valid dataset length: {len(self.datasets['valid'])}")
            log.info(f"Test dataset length: {len(self.datasets['test'])}")

        # load context recall dataset for eval
        if OmegaConf.select(self.cfg, "eval_context_recall", default=False) and self.cfg.context_recall_data_path is not None:
            _, ctx_recall_dset = hydra.utils.call(
                self.cfg.env.dataset,
                num_hist=self.cfg.num_hist,
                num_pred=self.cfg.num_pred,
                frameskip=self.cfg.frameskip,
                num_frames=self.cfg.ctx_recall_num_frames,
                data_path=self.cfg.context_recall_data_path,
                include_test=True,
            )
            self.context_recall_dset = {"valid": ctx_recall_dset["valid"], "test": ctx_recall_dset["test"]}
            if self.accelerator.is_main_process:
                log.info(f"Context recall valid dataset length: {len(self.context_recall_dset['valid'])}")
                log.info(f"Context recall test dataset length: {len(self.context_recall_dset['test'])}")

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]
        self.test_traj_dset = traj_dsets["test"]

        nw = self.cfg.num_workers
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=True,
                num_workers=nw if x == "train" else 1,
                collate_fn=None,
                pin_memory=True,
                multiprocessing_context=CTX,
                drop_last=True if x == "train" else False,
                prefetch_factor=2,
            )
            for x in ["train", "valid", "test"]
        }

        if self.accelerator.is_main_process:
            log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"], self.dataloaders["test"] = (
            self.accelerator.prepare(
                self.dataloaders["train"], self.dataloaders["valid"], self.dataloaders["test"]
            )
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(
            f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}"
        )

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()
        
        # Load checkpoint after all models are instantiated
        with self.accelerator.main_process_first():
            model_ckpt = (
                Path(self.cfg.saved_folder)
                / "checkpoints"
                / "model_latest.pth"
            )
            if model_ckpt.exists():
                self.load_ckpt(model_ckpt)
                log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")
        self.accelerator.wait_for_everyone()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]

            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(
                os.getcwd(), f"checkpoints/model_{self.epoch}.pth"
            )
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def save_ckpt_state_dict(self):
        """Save checkpoint using state dicts instead of full modules for safer DDP saving."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            ckpt = {}
            for k in self._keys_to_save:
                if k == "epoch":
                    ckpt[k] = self.__dict__[k]
                elif k.endswith("_optimizer"):
                    # Save optimizer state dict
                    if hasattr(self.__dict__[k], "state_dict"):
                        ckpt[k] = self.__dict__[k].state_dict()
                    else:
                        log.warning(
                            f"Optimizer {k} does not have state_dict method"
                        )
                else:
                    # Save model state dict
                    if hasattr(self.__dict__[k], "state_dict"):
                        if hasattr(self.__dict__[k], "module"):
                            # DDP wrapped model
                            ckpt[k] = self.accelerator.unwrap_model(
                                self.__dict__[k]
                            ).state_dict()
                        else:
                            ckpt[k] = self.__dict__[k].state_dict()
                    else:
                        log.warning(
                            f"Model {k} does not have state_dict method"
                        )

            # Save training config for reconstruction
            ckpt["train_cfg"] = self.cfg

            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model state dicts to {}".format(os.getcwd()))
            ckpt_path = os.path.join(
                os.getcwd(), f"checkpoints/model_{self.epoch}.pth"
            )
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        """Load checkpoint, handling both legacy full modules and new state dict format."""
        ckpt = torch.load(filename)

        # Check if this is a state dict format checkpoint
        if "train_cfg" in ckpt and any(
            k in ckpt
            for k in [
                "encoder",
                "predictor",
                "proprio_encoder",
                "action_encoder",
                "decoder",
            ]
        ):
            # This is a state dict format checkpoint
            log.info("Loading state dict format checkpoint")
            self._load_ckpt_state_dict(ckpt)
        else:
            # Legacy format - load full modules
            log.info("Loading legacy format checkpoint")
            for k, v in ckpt.items():
                self.__dict__[k] = v
            not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
            if len(not_in_ckpt):
                log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def _load_ckpt_state_dict(self, ckpt):
        """Load checkpoint from state dict format."""
        # Load epoch
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]

        # Load model state dicts
        for k in self._keys_to_save:
            if k == "epoch":
                continue
            elif k.endswith("_optimizer"):
                # Load optimizer state dict
                if k in ckpt and hasattr(self.__dict__[k], "load_state_dict"):
                    self.__dict__[k].load_state_dict(ckpt[k])
                    log.info(f"Loaded optimizer {k} from state dict")
                else:
                    log.warning(
                        f"Optimizer {k} not found in checkpoint or no load_state_dict method"
                    )
            else:
                # Load model state dict
                if k in ckpt and hasattr(self.__dict__[k], "load_state_dict"):
                    # Handle DDP wrapped models
                    if hasattr(self.__dict__[k], "module"):
                        self.accelerator.unwrap_model(
                            self.__dict__[k]
                        ).load_state_dict(ckpt[k])
                    else:
                        self.__dict__[k].load_state_dict(ckpt[k])
                    log.info(f"Loaded model {k} from state dict")
                else:
                    log.warning(
                        f"Model {k} not found in checkpoint or no load_state_dict method"
                    )

        # Check for missing keys
        model_keys = [
            k
            for k in self._keys_to_save
            if not k.endswith("_optimizer") and k != "epoch"
        ]
        not_in_ckpt = set(model_keys) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning(
                "Model keys not found in state dict ckpt: %s", not_in_ckpt
            )

    def init_models(self):

        # initialize encoder
        with self.accelerator.main_process_first():
            if self.encoder is None:
                self.encoder = hydra.utils.instantiate(
                    self.cfg.encoder,
                )
        self.accelerator.wait_for_everyone()

        # Sanity: make sure everyone actually built it and it has params
        n_params = sum(p.numel() for p in self.encoder.parameters())
        assert (
            n_params > 0
        ), "Encoder has zero parameters on this rank BEFORE DDP."

        if self.train_encoder:
            print("Freezing the first 9 transformer blocks")
            # freeze the first 9 transformer blocks
            for i, block in enumerate(self.encoder.base_model.blocks):
                if i < 9:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    # unfreeze the last 3 transformer blocks
                    for param in block.parameters():
                        param.requires_grad = True
            # unfreeze the layernorm
            for param in self.encoder.base_model.norm.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        # update cfg for saving
        with open_dict(self.cfg):
            self.cfg.proprio_encoder.emb_dim = self.proprio_encoder.emb_dim
            self.cfg.proprio_encoder.in_chans = self.datasets[
                "train"
            ].proprio_dim

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")
        # update cfg for saving
        with open_dict(self.cfg):
            self.cfg.action_encoder.emb_dim = self.action_encoder.emb_dim
            self.cfg.action_encoder.in_chans = self.datasets[
                "train"
            ].action_dim

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                dim = self.encoder.emb_dim + (
                    proprio_emb_dim * self.cfg.num_proprio_repeat
                    + action_emb_dim * self.cfg.num_action_repeat
                ) * (self.cfg.concat_dim)
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=dim,
                )
                # update cfg for saving
                with open_dict(self.cfg):
                    self.cfg.predictor.dim = dim
                    self.cfg.predictor.num_patches = num_patches
                    self.cfg.predictor.num_frames = self.cfg.num_hist

            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

            if self.accelerator.is_main_process:
                self.wandb_run.watch(self.predictor)

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
                    # update cfg for saving
                    with open_dict(self.cfg):
                        self.cfg.decoder.emb_dim = self.encoder.emb_dim
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False

        for name, mod in dict(
            encoder=self.encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio=self.proprio_encoder,
            action=self.action_encoder,
        ).items():
            if mod is not None:
                n = sum(p.numel() for p in mod.parameters())
                assert n >= 0, f"{name} has negative param count?"
        self.accelerator.wait_for_everyone()

        (
            self.encoder,
            self.predictor,
            self.decoder,
            self.proprio_encoder,
            self.action_encoder,
        ) = self.accelerator.prepare(
            self.encoder,
            self.predictor,
            self.decoder,
            self.proprio_encoder,
            self.action_encoder,
        )

        self.model = hydra.utils.instantiate(
            self.cfg.model,
            image_size=self.cfg.img_size,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            decoder_loss_type=getattr(
                self.cfg.training, "decoder_loss_type", "mse"
            ),
        )

        if self.accelerator.is_main_process:
            print(f"Model type: {type(self.model)}")
            print(self.model)

        

    def init_optimizers(self):
        # Scale learning rates by number of processes for proper multi-GPU training
        lr_scale = 1

        if self.cfg.training.encoder_weight_decay is None:
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.cfg.training.encoder_lr * lr_scale,
            )
        else:
            self.encoder_optimizer = torch.optim.AdamW(
                self.encoder.parameters(),
                lr=self.cfg.training.encoder_lr * lr_scale,
                weight_decay=self.cfg.training.encoder_weight_decay,
            )
        self.encoder_optimizer = self.accelerator.prepare(
            self.encoder_optimizer
        )
        if self.accelerator.is_main_process:
            log.info(
                f"Scaling learning rates by {lr_scale} for {self.accelerator.num_processes} processes"
            )
            log.info(
                f"Encoder LR: {self.cfg.training.encoder_lr} -> {self.cfg.training.encoder_lr * lr_scale}"
            )

        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr * lr_scale,
                weight_decay=self.cfg.training.predictor_weight_decay,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )
            if self.accelerator.is_main_process:
                log.info(
                    f"Predictor LR: {self.cfg.training.predictor_lr} -> {self.cfg.training.predictor_lr * lr_scale}"
                )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(),
                    self.proprio_encoder.parameters(),
                ),
                lr=self.cfg.training.action_encoder_lr * lr_scale,
                weight_decay=self.cfg.training.action_encoder_weight_decay,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )
            if self.accelerator.is_main_process:
                log.info(
                    f"Action Encoder LR: {self.cfg.training.action_encoder_lr} -> {self.cfg.training.action_encoder_lr * lr_scale}"
                )

        if self.cfg.has_decoder:
            if self.cfg.training.decoder_weight_decay is None:
                self.decoder_optimizer = torch.optim.Adam(
                    self.decoder.parameters(),
                    lr=self.cfg.training.decoder_lr * lr_scale,
                )
            else:
                self.decoder_optimizer = torch.optim.AdamW(
                    self.decoder.parameters(),
                    lr=self.cfg.training.decoder_lr * lr_scale,
                    weight_decay=self.cfg.training.decoder_weight_decay,
                )

            self.decoder_optimizer = self.accelerator.prepare(
                self.decoder_optimizer
            )
            if self.accelerator.is_main_process:
                log.info(
                    f"Decoder LR: {self.cfg.training.decoder_lr} -> {self.cfg.training.decoder_lr * lr_scale}"
                )

        # Initialize learning rate schedulers
        self.schedulers = {}
        if self.cfg.training.use_scheduler:
            T_0 = self.cfg.training.T_0
            T_mult = self.cfg.training.T_mult
            eta_min_ratio = self.cfg.training.eta_min_ratio
            decay_factor = self.cfg.training.decay_factor

            # Calculate steps per epoch for step-based scheduling
            steps_per_window = 1 + (self.cfg.num_frames - self.window_size) // self.step_size
            steps_per_epoch = len(self.dataloaders["train"]) * steps_per_window
            warmup_steps = int(
                self.cfg.training.warmup_percent * steps_per_epoch
            )
            T_0_steps = T_0 * steps_per_epoch  # Convert epochs to steps
            if self.accelerator.is_main_process:
                log.info(
                    f"Lr scheduler: {steps_per_epoch} steps per epoch, T_0={T_0_steps} steps, warmup={warmup_steps} steps"
                )

            # Encoder scheduler
            self.schedulers["encoder"] = CosineAnnealingWarmRestartsDecay(
                self.encoder_optimizer,
                T_0=T_0_steps,
                T_mult=T_mult,
                eta_min=self.cfg.training.encoder_lr
                * lr_scale
                * eta_min_ratio,
                decay_factor=decay_factor,
                warmup_epochs=warmup_steps,
            )

            if self.cfg.has_predictor:
                # Predictor scheduler
                self.schedulers["predictor"] = (
                    CosineAnnealingWarmRestartsDecay(
                        self.predictor_optimizer,
                        T_0=T_0_steps,
                        T_mult=T_mult,
                        eta_min=self.cfg.training.predictor_lr
                        * lr_scale
                        * eta_min_ratio,
                        decay_factor=decay_factor,
                        warmup_epochs=warmup_steps,
                    )
                )

                # Action encoder scheduler
                self.schedulers["action_encoder"] = (
                    CosineAnnealingWarmRestartsDecay(
                        self.action_encoder_optimizer,
                        T_0=T_0_steps,
                        T_mult=T_mult,
                        eta_min=self.cfg.training.action_encoder_lr
                        * lr_scale
                        * eta_min_ratio,
                        decay_factor=decay_factor,
                        warmup_epochs=warmup_steps,
                    )
                )

            if self.cfg.has_decoder:
                # Decoder scheduler
                self.schedulers["decoder"] = CosineAnnealingWarmRestartsDecay(
                    self.decoder_optimizer,
                    T_0=T_0_steps,
                    T_mult=T_mult,
                    eta_min=self.cfg.training.decoder_lr
                    * lr_scale
                    * eta_min_ratio,
                    decay_factor=decay_factor,
                    warmup_epochs=warmup_steps,
                )

            if self.accelerator.is_main_process:
                log.info(
                    f"Initialized step-based cosine LR schedulers: T_0={T_0_steps} steps, T_mult={T_mult}, eta_min_ratio={eta_min_ratio}, decay_factor={decay_factor}, warmup_steps={warmup_steps}"
                )

    def run(self):

        init_epoch = self.epoch + 1  # epoch starts from 1
        if self.cfg.dry_run:
            self.total_epochs = 1
        self.train_steps = 0
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()

            # Start timing the epoch
            epoch_start_time = time.time()

            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.accelerator.wait_for_everyone()

            # Calculate epoch execution time
            epoch_time = time.time() - epoch_start_time

            # log process memory usage
            mems = get_all_process_memory_mb()
            mems_log = {f"mem_{k}": [v] for k, v in mems.items()}
            self.logs_update(mems_log)

            # Add epoch time to logs before flashing
            epoch_time_log = {"epoch_time": [epoch_time]}
            self.logs_update(epoch_time_log)

            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                # ckpt_path, model_name, model_epoch = self.save_ckpt()
                ckpt_path, model_name, model_epoch = (
                    self.save_ckpt_state_dict()
                )

            if self.cfg.dry_run:
                break

        # test and save results
        self.test()
        self.accelerator.wait_for_everyone()

    def train_step(self, obs, act):
        self.model.train()
        self.encoder_optimizer.zero_grad()
        if self.cfg.has_decoder:
            self.decoder_optimizer.zero_grad()
        if self.cfg.has_predictor:
            self.predictor_optimizer.zero_grad()
            self.action_encoder_optimizer.zero_grad()

        z_out, visual_out, visual_reconstructed, loss, loss_components = (
            self.model(obs, act)
        )
        self.accelerator.backward(loss)
        assert not torch.isnan(loss), f"Loss is NaN at epoch {self.epoch}"

        # Gradient norm clipping
        grad_norms = {}
        if self.model.train_encoder:
            grad_norms["encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.cfg.training.max_grad_norm
            ).item()
        if self.cfg.has_predictor and self.model.train_predictor:
            grad_norms["predictor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.predictor.parameters(), self.cfg.training.max_grad_norm
            ).item()
            grad_norms["action_encoder_grad_norm"] = (
                torch.nn.utils.clip_grad_norm_(
                    self.action_encoder.parameters(),
                    self.cfg.training.max_grad_norm,
                ).item()
            )
            grad_norms["proprio_encoder_grad_norm"] = (
                torch.nn.utils.clip_grad_norm_(
                    self.proprio_encoder.parameters(),
                    self.cfg.training.max_grad_norm,
                ).item()
            )
        if self.cfg.has_decoder and self.model.train_decoder:
            grad_norms["decoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), self.cfg.training.max_grad_norm
            ).item()

        if self.accelerator.is_main_process:
            for name, norm in grad_norms.items():
                self.logs_update({f"train_{name}": [norm]})

        if self.model.train_encoder:
            self.encoder_optimizer.step()
        if self.cfg.has_decoder and self.model.train_decoder:
            self.decoder_optimizer.step()
        if self.cfg.has_predictor and self.model.train_predictor:
            self.predictor_optimizer.step()
            self.action_encoder_optimizer.step()

        # # Step learning rate schedulers per batch if step-based scheduling is enabled
        if self.cfg.training.use_scheduler:
            for scheduler in self.schedulers.values():
                scheduler.step()

        loss = self.accelerator.gather_for_metrics(loss).mean()

        loss_components = self.accelerator.gather_for_metrics(loss_components)
        loss_components = {
            key: value.mean().item() for key, value in loss_components.items()
        }

        z_components = {
            "z_out": z_out,
            "visual_out": visual_out,
            "visual_reconstructed": visual_reconstructed,
        }

        return loss_components, z_components

    def train(self):

        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        prev_time = time.perf_counter()

        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):

            if hasattr(self.model.predictor, "reset_memory"):
                self.model.predictor.reset_memory()
            elif hasattr(self.model.predictor, "module") and hasattr(
                self.model.predictor.module, "reset_memory"
            ):
                self.model.predictor.module.reset_memory()

            batch_loss_components = defaultdict(float)

            data_time = time.perf_counter() - prev_time
            obs, act, _ = data
            B, N = obs["visual"].shape[:2]
            num_windows = max(1, 1 + (N - self.window_size) // self.step_size)

            plot = i == 0
            self.model.train()
            compute_start.record()

            for window_idx in range(num_windows):
                start_idx = window_idx * self.step_size
                end_idx = min(start_idx + self.window_size, N)

                obs_window = {
                    k: v[:, start_idx:end_idx, ...] for k, v in obs.items()
                }
                act_window = act[:, start_idx:end_idx, ...]
                loss_components, z_components = self.train_step(
                    obs_window, act_window
                )
                for k, v in loss_components.items():
                    batch_loss_components[k] += v / num_windows
                self.train_steps += 1

            compute_end.record()
            torch.cuda.synchronize(self.accelerator.device)
            compute_ms = compute_start.elapsed_time(compute_end)

            if i < 5 or i % 100 == 0:
                time_logs = {
                    "train_data_time": [data_time],
                    "train_compute_time": [compute_ms / 1e3],
                    "train_compute_time_per_window": [
                        compute_ms / 1e3 / num_windows
                    ],
                    "train_batch_size": [B],
                    "train_num_windows": [num_windows],
                    "train_window_size": [self.window_size],
                    "train_num_frames": [N],
                }
                self.logs_update(time_logs)

            if self.cfg.has_decoder and plot:
                self.decoder_eval(i, obs_window, z_components)

            self.logs_update(
                {f"train_{k}": [v] for k, v in batch_loss_components.items()}
            )

            if (
                self.cfg.has_predictor and i % 100 == 0
            ):  # Log every 100 batches
                alpha_logs = self.get_alpha_values()
                self.logs_update(
                    {f"train_{k}": [v] for k, v in alpha_logs.items()}
                )

            prev_time = time.perf_counter()

            if self.cfg.dry_run:
                break

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            rand_start_end = (
                False if self.cfg.env == "deformable_env" else True
            )
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset,
                    mode="train",
                    rand_start_end=rand_start_end,
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(
                    self.val_traj_dset,
                    mode="val",
                    rand_start_end=rand_start_end,
                )
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

                if self.epoch % self.cfg.eval_every_x_epoch == 0:
                    # long horizon treatments
                    if OmegaConf.select(self.cfg, "horizon_treatment", default=None) is not None:
                        train_long_horizon_logs = self.openloop_rollout(
                            self.train_traj_dset,
                            mode="train",
                            rand_start_end=False,
                            horizon_treatment=self.cfg.horizon_treatment,
                        )
                        train_long_horizon_logs = {
                            f"train_{k}": [v] for k, v in train_long_horizon_logs.items()
                        }
                        self.logs_update(train_long_horizon_logs)

                        val_long_horizon_logs = self.openloop_rollout(
                            self.val_traj_dset,
                            mode="val",
                            rand_start_end=False,
                            horizon_treatment=self.cfg.horizon_treatment,
                        )
                        val_long_horizon_logs = {
                            f"val_{k}": [v] for k, v in val_long_horizon_logs.items()
                        }
                        self.logs_update(val_long_horizon_logs)

                    # loop closure tests
                    if OmegaConf.select(self.cfg, "eval_loopclosure", default=None) is not None and self.cfg.eval_loopclosure.data_path is not None:
                        from datasets.minigrid_dataset import MiniGridMemmapDataset
                        loop_dset = MiniGridMemmapDataset(
                            n_rollout=None,
                            transform=self.cfg.dataset.transform,
                            data_path=self.cfg.eval_loopclosure.data_path,
                            normalize_action=False,
                            total_episodes=self.cfg.eval_loopclosure.total_episodes,
                            proprio_available=True,
                        )
                        loop_rollout_logs = self.loopclosure_rollout(loop_dset)
                        loop_rollout_logs = {
                            f"val_{k}": [v] for k, v in loop_rollout_logs.items()
                        }
                        self.logs_update(loop_rollout_logs)

                    # long imagination
                    if OmegaConf.select(self.cfg, "eval_long_imagination", default=False):
                        long_imagination_logs = self.long_imagination_rollout(self.val_traj_dset, query_phase_start_idx=self.cfg.query_phase_start_idx, num_rollout=self.cfg.num_eval_samples)
                        long_imagination_logs = {
                            f"val_{k}": [v] for k, v in long_imagination_logs.items()
                        }
                        self.logs_update(long_imagination_logs)

                    # context recall
                    if OmegaConf.select(self.cfg, "eval_context_recall", default=False) and self.context_recall_dset is not None:
                        context_recall_logs = self.context_recall_rollout(self.context_recall_dset["valid"], query_phase_start_idx=self.cfg.teleport_start_idx, num_rollout=self.cfg.num_eval_samples)
                        context_recall_logs = {
                            f"val_{k}": [v] for k, v in context_recall_logs.items()
                        }
                        self.logs_update(context_recall_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            batch_loss_components = defaultdict(float)
            obs, act, _ = data
            B, N = obs["visual"].shape[:2]
            num_windows = max(1, 1 + (N - self.window_size) // self.step_size)

            if hasattr(self.model.predictor, "reset_memory"):
                self.model.predictor.reset_memory()
            elif hasattr(self.model.predictor, "module") and hasattr(
                self.model.predictor.module, "reset_memory"
            ):
                self.model.predictor.module.reset_memory()

            for window_idx in range(num_windows):
                start_idx = window_idx * self.step_size
                end_idx = min(start_idx + self.window_size, N)
                obs_window = {
                    k: v[:, start_idx:end_idx, ...] for k, v in obs.items()
                }
                act_window = act[:, start_idx:end_idx, ...]

                # val step
                with torch.no_grad():
                    (
                        z_out,
                        visual_out,
                        visual_reconstructed,
                        loss,
                        loss_components,
                    ) = self.model(obs_window, act_window)
                    loss = self.accelerator.gather_for_metrics(loss).mean()
                    loss_components = self.accelerator.gather_for_metrics(
                        loss_components
                    )
                    for key, value in loss_components.items():
                        batch_loss_components[key] += (
                            value.mean().item() / num_windows
                        )

            if self.cfg.has_decoder and i == 0:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, _ = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs_window)
                    z_tgt = slice_trajdict_with_t(
                        z_gt, start_idx=self.cfg.num_pred
                    )

                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item()
                        for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist,
                        self.cfg.num_hist + self.cfg.num_pred,
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred],
                            obs_window["visual"][:, t],
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs_window["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t],
                            obs_window["visual"][:, t],
                        )
                        img_reconstruction_scores = (
                            self.accelerator.gather_for_metrics(
                                img_reconstruction_scores
                            )
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs_window["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )

            self.logs_update(
                {f"val_{k}": [v] for k, v in batch_loss_components.items()}
            )

            if (
                self.cfg.predictor == "additive_control_vit"
                and self.cfg.has_predictor
                and i == 0
            ):  # Log on first validation batch
                alpha_logs = self.get_alpha_values()
                self.logs_update(
                    {f"val_{k}": [v] for k, v in alpha_logs.items()}
                )

            if self.cfg.dry_run:
                break

    def _safe_convert_to_numpy(self, value):
        """Safely convert tensor values to numpy for CSV storage."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            return [self._safe_convert_to_numpy(v) for v in value]
        else:
            return value

    def test(self):
        """Test function that mimics val() but uses test dataset and saves results to CSV."""
        self.model.eval()

        print("Running eval on test set...")

        if len(self.test_traj_dset) > 0 and self.cfg.has_predictor:
            rand_start_end = (
                False if self.cfg.env == "deformable_env" else True
            )
            with torch.no_grad():
                test_rollout_logs = self.openloop_rollout(
                    self.test_traj_dset,
                    mode="test",
                    rand_start_end=rand_start_end,
                )
                test_rollout_logs = {
                    f"test_{k}": [v] for k, v in test_rollout_logs.items()
                }
                self.logs_update(test_rollout_logs)

                # long horizon treatments
                if OmegaConf.select(self.cfg, "horizon_treatment", default=None) is not None:
                    test_long_horizon_logs = self.openloop_rollout(
                        self.test_traj_dset,
                        mode="test",
                        rand_start_end=False,
                        horizon_treatment=self.cfg.horizon_treatment,
                    )
                    test_long_horizon_logs = {
                        f"test_{k}": [v] for k, v in test_long_horizon_logs.items()
                    }
                    self.logs_update(test_long_horizon_logs)

                # long imagination
                if OmegaConf.select(self.cfg, "eval_long_imagination", default=False):
                    # run on full test dataset
                    long_imagination_logs = self.long_imagination_rollout(
                        self.test_traj_dset, 
                        query_phase_start_idx=self.cfg.query_phase_start_idx, 
                        num_rollout=self.cfg.num_eval_samples if self.cfg.dry_run else None,
                        plotting_dir=f"test_results/e{self.epoch}_long_imagination"
                    )
                    long_imagination_logs = {
                        f"test_{k}": [v] for k, v in long_imagination_logs.items()
                    }
                    self.logs_update(long_imagination_logs)

                # context recall
                if (
                    OmegaConf.select(
                        self.cfg, "eval_context_recall", default=False
                    )
                    and self.context_recall_dset is not None
                ):
                    context_recall_logs = self.context_recall_rollout(
                        self.context_recall_dset["test"],
                        query_phase_start_idx=self.cfg.teleport_start_idx,
                        num_rollout=self.cfg.num_eval_samples if self.cfg.dry_run else None,
                        plotting_dir=f"test_results/e{self.epoch}_context_recall"
                    )
                    context_recall_logs = {
                        f"test_{k}": [v] for k, v in context_recall_logs.items()
                    }
                    self.logs_update(context_recall_logs)

        self.accelerator.wait_for_everyone()

        # Test on test dataloader
        for i, data in enumerate(
            tqdm(self.dataloaders["test"], desc=f"Epoch {self.epoch} Test")
        ):
            batch_loss_components = defaultdict(float)
            obs, act, _ = data
            B, N = obs["visual"].shape[:2]
            num_windows = max(1, 1 + (N - self.window_size) // self.step_size)

            if hasattr(self.model.predictor, "reset_memory"):
                self.model.predictor.reset_memory()
            elif hasattr(self.model.predictor, "module") and hasattr(
                self.model.predictor.module, "reset_memory"
            ):
                self.model.predictor.module.reset_memory()

            for window_idx in range(num_windows):
                start_idx = window_idx * self.step_size
                end_idx = min(start_idx + self.window_size, N)
                obs_window = {
                    k: v[:, start_idx:end_idx, ...] for k, v in obs.items()
                }
                act_window = act[:, start_idx:end_idx, ...]

                # test step
                with torch.no_grad():
                    (
                        z_out,
                        visual_out,
                        visual_reconstructed,
                        loss,
                        loss_components,
                    ) = self.model(obs_window, act_window)
                    loss = self.accelerator.gather_for_metrics(loss).mean()
                    loss_components = self.accelerator.gather_for_metrics(
                        loss_components
                    )
                    for key, value in loss_components.items():
                        batch_loss_components[key] += (
                            value.mean().item() / num_windows
                        )

            if self.cfg.has_decoder and i == 0:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, _ = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs_window)
                    z_tgt = slice_trajdict_with_t(
                        z_gt, start_idx=self.cfg.num_pred
                    )

                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item()
                        for key, value in err_logs.items()
                    }
                    err_logs = {f"test_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                    # Store error logs for CSV
                    # for k, v in err_logs.items():
                    #     test_results[k].extend(self._safe_convert_to_numpy(v))

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist,
                        self.cfg.num_hist + self.cfg.num_pred,
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred],
                            obs_window["visual"][:, t],
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"test_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                        # Store image prediction scores for CSV
                        # for k, v in img_pred_scores.items():
                        #     test_results[k].extend(self._safe_convert_to_numpy(v))

                if visual_reconstructed is not None:
                    for t in range(obs_window["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t],
                            obs_window["visual"][:, t],
                        )
                        img_reconstruction_scores = (
                            self.accelerator.gather_for_metrics(
                                img_reconstruction_scores
                            )
                        )
                        img_reconstruction_scores = {
                            f"test_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                        # Store image reconstruction scores for CSV
                        # for k, v in img_reconstruction_scores.items():
                        #     test_results[k].extend(self._safe_convert_to_numpy(v))

                self.plot_samples(
                    obs_window["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="test",
                )

            # Store batch loss components for CSV
            batch_loss_logs = {f"test_{k}": [v] for k, v in batch_loss_components.items()}
            self.logs_update(batch_loss_logs)
            # for k, v in batch_loss_logs.items():
            #     test_results[k].extend(self._safe_convert_to_numpy(v))

            if (
                self.cfg.predictor == "additive_control_vit"
                and self.cfg.has_predictor
                and i == 0
            ):  # Log on first test batch
                alpha_logs = self.get_alpha_values()
                alpha_logs = {f"test_{k}": [v] for k, v in alpha_logs.items()}
                self.logs_update(alpha_logs)

                # # Store alpha logs for CSV
                # for k, v in alpha_logs.items():
                #     test_results[k].extend(self._safe_convert_to_numpy(v))

            if self.cfg.dry_run:
                break

        # # Save test results to CSV
        # if self.accelerator.is_main_process and test_results:
        #     self.save_test_results_to_csv(test_results)

    def save_test_results_to_csv(self, test_results):
        """Save test results to CSV file."""
        if not test_results:
            return

        try:
            # Create test results directory
            test_dir = "test_results"
            os.makedirs(test_dir, exist_ok=True)    

            # Prepare CSV data
            csv_data = []
            max_length = max(len(v) for v in test_results.values()) if test_results else 0

            for i in range(max_length):
                row = {"epoch": self.epoch, "sample_idx": i}
                for key, values in test_results.items():
                    row[key] = values[i] if i < len(values) else None
                csv_data.append(row)

            # Write to CSV
            csv_filename = f"{test_dir}/test_results_epoch_{self.epoch}.csv"
            fieldnames = ["epoch", "sample_idx"] + list(test_results.keys())

            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            log.info(f"Test results saved to {csv_filename}")

            # Also save summary statistics
            summary_filename = f"{test_dir}/test_summary_epoch_{self.epoch}.csv"
            summary_data = []
            for key, values in test_results.items():
                if values:  # Only process non-empty lists
                    summary_data.append({
                        "metric": key,
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    })

            if summary_data:
                with open(summary_filename, "w", newline="") as csvfile:
                    fieldnames = ["metric", "mean", "std", "min", "max", "count"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_data)

                log.info(f"Test summary saved to {summary_filename}")

        except Exception as e:
            log.error(f"Error saving test results to CSV: {e}")

    def horizon_treatment_eval(
        self, z_pred, z_tgt, obs_tgt, obs_recon, reencoded_visuals
    ):
        logs = {}
        for k in z_pred.keys():
            # mse between z_pred and z_tgt latents
            logs[f"{k}_latent_mse"] = torch.nn.functional.mse_loss(
                z_pred[k], z_tgt[k]
            )

            # l2 between z_pred and z_tgt latents
            logs[f"{k}_latent_l2"] = torch.norm(z_pred[k] - z_tgt[k], p=2)

            # cycle consistency (decode > encode > measure)
            logs[f"{k}_latent_cycle_mse"] = torch.nn.functional.mse_loss(
                reencoded_visuals[k], z_tgt[k]
            )
            logs[f"{k}_latent_cycle_l2"] = torch.norm(
                reencoded_visuals[k] - z_tgt[k], p=2
            )

            # mse between obs_recon and obs_tgt
            logs[f"{k}_obs_recon_mse"] = torch.nn.functional.mse_loss(
                obs_recon[k], obs_tgt[k]
            )
            logs[f"{k}_obs_recon_l2"] = torch.norm(
                obs_recon[k] - obs_tgt[k], p=2
            )

        return logs

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.cfg.num_pred, None),
            "next1": (-self.cfg.num_pred, -self.cfg.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def decoder_eval(self, batch_idx, obs, z_components):
        # only eval images when plotting due to speed
        if self.cfg.has_predictor:
            z_obs_out, _ = self.model.separate_emb(z_components["z_out"])
            z_gt = self.model.encode_obs(obs)
            z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.cfg.num_pred)

            # state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
            err_logs = self.err_eval(z_obs_out, z_tgt)

            err_logs = self.accelerator.gather_for_metrics(err_logs)
            err_logs = {
                key: value.mean().item() for key, value in err_logs.items()
            }
            err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

            self.logs_update(err_logs)

        if z_components["visual_out"] is not None:
            for t in range(
                self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
            ):
                img_pred_scores = eval_images(
                    z_components["visual_out"][:, t - self.cfg.num_pred],
                    obs["visual"][:, t],
                )
                img_pred_scores = self.accelerator.gather_for_metrics(
                    img_pred_scores
                )
                img_pred_scores = {
                    f"train_img_{k}_pred": [v.mean().item()]
                    for k, v in img_pred_scores.items()
                }
                self.logs_update(img_pred_scores)

        if z_components["visual_reconstructed"] is not None:
            for t in range(obs["visual"].shape[1]):
                img_reconstruction_scores = eval_images(
                    z_components["visual_reconstructed"][:, t],
                    obs["visual"][:, t],
                )
                img_reconstruction_scores = (
                    self.accelerator.gather_for_metrics(
                        img_reconstruction_scores
                    )
                )
                img_reconstruction_scores = {
                    f"train_img_{k}_reconstructed": [v.mean().item()]
                    for k, v in img_reconstruction_scores.items()
                }
                self.logs_update(img_reconstruction_scores)

        if self.accelerator.is_main_process:
            self.plot_samples(
                obs["visual"],
                z_components["visual_out"],
                z_components["visual_reconstructed"],
                self.epoch,
                batch=batch_idx,
                num_samples=self.num_reconstruct_samples,
                phase="train",
            )

    def openloop_rollout(
        self,
        dset,
        num_rollout=10,
        rand_start_end=True,
        min_horizon=2,
        mode="train",
        horizon_treatment=None,
    ):
        if horizon_treatment is not None:
            T = dset[0][0]["visual"].shape[0] // self.cfg.frameskip
            assert horizon_treatment < T, "horizon_treatment must be less than T"

        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = defaultdict(list)

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        if hasattr(self.model.predictor, "reset_memory"):
            self.model.predictor.reset_memory()
        elif hasattr(self.model.predictor, "module") and hasattr(
            self.model.predictor.module, "reset_memory"
        ):
            self.model.predictor.module.reset_memory()

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end and horizon_treatment is None:
                    if (
                        obs["visual"].shape[0]
                        > min_horizon * self.cfg.frameskip + 1
                    ):
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0]
                            - min_horizon * self.cfg.frameskip
                            - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (
                        obs["visual"].shape[0] - start - 1
                    ) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(
                            min_horizon, max_horizon + 1
                        )
                elif horizon_treatment is not None:
                    start = 0
                    horizon = horizon_treatment
                    valid_traj = True
                else:
                    valid_traj = True
                    start = 0
                    horizon = (
                        obs["visual"].shape[0] - 1
                    ) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : start
                    + horizon * self.cfg.frameskip
                    + 1 : self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)
                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(
                    z_obses, start_idx=-1, end_idx=None
                )
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    log_key += f"_h{horizon}" if horizon_treatment is not None else ""
                    logs[log_key].append(div_loss[k].cpu().numpy())

                if self.cfg.has_decoder:
                    decoded = self.model.decode_obs(z_obses)[0]
                    visuals = decoded["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    if self.accelerator.is_main_process:
                        self.plot_imgs(
                            imgs,
                            obs["visual"].shape[0],
                            f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}_h{horizon}.png",
                        )           

                if horizon_treatment is not None:
                    # compute rollout error progression
                    obs_tgt = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}
                    z_tgts = self.model.encode_obs(obs_tgt)
                    z_cycle = self.model.encode_obs({"visual": visuals, "proprio": obs_tgt["proprio"]}) # re-encode the decoded visuals; use proprio from obs instead of decoded
                    div_loss = self.horizon_treatment_eval(z_obses, z_tgts, obs_tgt, {"visual": visuals, "proprio": obs_tgt["proprio"]}, z_cycle)
                    for k in div_loss.keys():
                        logs[f"{k}_err_horizon_{postfix}_h{horizon}"].append(div_loss[k].cpu().numpy())

    
        for k, v in logs.items():
            logs[k] = np.mean(v) if v else 0

        return logs

    # minigrid only for now
    def loopclosure_rollout(self, dset, query_phase_start=0):
        plotting_dir = f"loopclosure_plots/e{self.epoch}_loopclosure"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = defaultdict(list)

        for idx in range(len(dset)):
            obs, act, _, _ = dset[idx]
            obs = {k:v.to(self.device) for k, v in obs.items()}
            act = act.to(self.device)
            actions = act.unsqueeze(0)

            horizon = int(actions.shape[1])

            obs_0 = {}
            for k in obs.keys():
                obs_0[k] = (
                    obs[k][:self.cfg.num_hist].unsqueeze(0).to(self.device)
                )  # unsqueeze for batch, (b, t, c, h, w)
            z_obses, _ = self.model.rollout(obs_0, actions)

            if self.cfg.has_decoder:
                decoded = self.model.decode_obs(z_obses)[0]
                visuals = decoded["visual"]
                imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                if self.accelerator.is_main_process:
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{idx}_h{horizon}.png",
                    )

                # compute rollout error progression
                obs_tgt = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}
                z_tgts = self.model.encode_obs(obs_tgt)
                z_cycle = self.model.encode_obs({"visual": visuals, "proprio": obs_tgt["proprio"]}) # re-encode the decoded visuals; use proprio from obs instead of decoded
                div_loss = self.horizon_treatment_eval(z_obses, z_tgts, obs_tgt, visuals, z_cycle)
                for k in div_loss.keys():
                    logs[f"z_{k}_err_loopclosure_h{horizon}"].append(div_loss[k].cpu().numpy())

        for k, v in logs.items():
            logs[k] = np.mean(v)

        return logs

    def long_imagination_rollout(self, dset, query_phase_start_idx=0, num_rollout=None, plotting_dir=None):
        if plotting_dir is None:
            plotting_dir = f"long_imagination_plots/e{self.epoch}_long_imagination"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        if num_rollout is None:
            num_rollout = len(dset)

        for idx in range(num_rollout):
            local_logs = defaultdict(list)
            obs, actions, _, _ = dset[idx]
            obs = {k:v.unsqueeze(0).to(self.device) for k, v in obs.items()}
            actions = actions.unsqueeze(0).to(self.device)

            # context phase
            num_ctx_windows = 1 + (query_phase_start_idx - self.window_size) // self.step_size
            for window_idx in range(num_ctx_windows):
                start_idx = window_idx * self.step_size
                end_idx = min(start_idx + self.window_size, query_phase_start_idx)
                obs_window = {
                    k: v[:, start_idx:end_idx] for k, v in obs.items()
                }
                act_window = actions[:, start_idx:end_idx]

                # burn in for context phase
                self.model(obs_window, act_window) # no tracking until query phase

            init_context = self.cfg.num_hist
            num_phase_steps = actions.shape[1] - query_phase_start_idx
            # rollout on query phase
            query_actions = actions[:, query_phase_start_idx-init_context:]
            obs_query_start = {
                k: v[
                    :, query_phase_start_idx - init_context : query_phase_start_idx
                ]
                for k, v in obs.items()
            }

            obs_tgt = {k: v[:, -num_phase_steps:] for k, v in obs.items()}
            z_tgts = self.model.encode_obs(obs_tgt)

            z_obses, _ = self.model.rollout(obs_query_start, query_actions, bypass_memory_reset=True)

            # evaluate on query phase
            decoded = self.model.decode_obs(z_obses)[0]
            decoded_tgt = self.model.decode_obs(z_tgts)[0]

            # eval only query phase
            visuals = decoded["visual"][:, -(num_phase_steps + 1): -1] # offset by 1 to exclude the last predicted frame which has no gt
            z_obses = {k: v[:, -(num_phase_steps + 1): -1] for k, v in z_obses.items()}

            if idx < min(10, num_rollout):
                imgs = torch.cat(
                    [
                        obs["visual"][0, -num_phase_steps:].cpu(), # this doesn't have the extra frame
                        decoded_tgt["visual"][0, -num_phase_steps:].cpu(),
                        visuals[0].cpu(),
                    ],
                    dim=0,
                )
                if self.accelerator.is_main_process:
                    self.plot_imgs(
                        imgs,
                        visuals.shape[1],
                        f"{plotting_dir}/e{self.epoch}_{idx}_long_imagination.png",
                    )     

            
            z_cycle = self.model.encode_obs({"visual": visuals, "proprio": obs_tgt["proprio"]}) # re-encode the decoded visuals; use proprio from obs instead of decoded
            div_loss = self.horizon_treatment_eval(z_obses, z_tgts, obs_tgt, {"visual": visuals, "proprio": obs_tgt["proprio"]}, z_cycle)
            for k in div_loss.keys():
                local_logs[f"{k}_err_long_imagination"].append(div_loss[k].cpu().numpy())

        # aggregate errors for each time step over rollouts
        for k, v in local_logs.items():
            logs[k] = np.mean(v)

        return logs

    def context_recall_rollout(self, dset, query_phase_start_idx=0, num_rollout=None, plotting_dir=None):
        if plotting_dir is None:
            plotting_dir = f"context_recall_plots/e{self.epoch}_context_recall"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}
        burn_in_steps = [0, 5, 10, 15, 20]

        if num_rollout is None:
            num_rollout = len(dset)

        for burn_in_step in burn_in_steps:
            local_query_phase_start_idx = query_phase_start_idx + burn_in_step

            for idx in range(num_rollout):
                local_logs = defaultdict(list)
                obs, actions, _, _ = dset[idx]
                obs = {k:v.unsqueeze(0).to(self.device) for k, v in obs.items()}
                actions = actions.unsqueeze(0).to(self.device)

                # context phase
                num_ctx_windows = 1 + (local_query_phase_start_idx - self.window_size) // self.step_size
                for window_idx in range(num_ctx_windows):
                    start_idx = window_idx * self.step_size
                    end_idx = min(start_idx + self.window_size, local_query_phase_start_idx)
                    obs_window = {
                        k: v[:, start_idx:end_idx] for k, v in obs.items()
                    }
                    act_window = actions[:, start_idx:end_idx]

                    # burn in for context phase
                    self.model(obs_window, act_window) # no tracking until query phase

                init_context = self.cfg.num_hist
                num_phase_steps = actions.shape[1] - local_query_phase_start_idx
                # rollout on query phase
                query_actions = actions[:, local_query_phase_start_idx-init_context:]
                obs_query_start = {
                    k: v[
                        :, local_query_phase_start_idx - init_context : local_query_phase_start_idx
                    ]
                    for k, v in obs.items()
                }

                z_obses, _ = self.model.rollout(obs_query_start, query_actions, bypass_memory_reset=True)

                obs_tgt = {k: v[:, -num_phase_steps:] for k, v in obs.items()}
                z_tgts = self.model.encode_obs(obs_tgt)

                # evaluate on query phase
                decoded = self.model.decode_obs(z_obses)[0]
                decoded_tgt = self.model.decode_obs(z_tgts)[0]
                
                # eval only query phase
                visuals = decoded["visual"][:, -(num_phase_steps + 1): -1] # offset by 1 to exclude the last predicted frame which has no gt
                z_obses = {k: v[:, -(num_phase_steps + 1): -1] for k, v in z_obses.items()}

                # save plots
                if idx < min(10, num_rollout):
                    imgs = torch.cat(
                        [
                            obs["visual"][0, -num_phase_steps:].cpu(), # this doesn't have the extra frame
                            decoded_tgt["visual"][0, -num_phase_steps:].cpu(),
                            visuals[0].cpu(),
                        ],
                        dim=0,
                    )
                    if self.accelerator.is_main_process:
                        self.plot_imgs(
                            imgs,
                            visuals.shape[1],
                            f"{plotting_dir}/e{self.epoch}_{idx}_context_recall_burn_in_{burn_in_step}.png",
                        )     

                z_cycle = self.model.encode_obs({"visual": visuals, "proprio": obs_tgt["proprio"]}) # re-encode the decoded visuals; use proprio from obs instead of decoded
                div_loss = self.horizon_treatment_eval(z_obses, z_tgts, obs_tgt, {"visual": visuals, "proprio": obs_tgt["proprio"]}, z_cycle)
                for k in div_loss.keys():
                    local_logs[f"{k}_err_context_recall_burn_in_{burn_in_step}"].append(div_loss[k].cpu().numpy())
            
            # aggregate over rollouts
            for k, v in local_logs.items():
                logs[k] = np.mean(v)

        return logs

    def oracle_memory_benchmark(self, dset, oracle_mode='perfect_memory', num_rollouts=2, prepend_type='half'):
        if plotting_dir is None:
            plotting_dir = f"oracle_{oracle_mode}/e{self.epoch}_oracle_{oracle_mode}"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # for a given context length, burn in num_hist context frames
        # encode next window of frames as the memory latents
        # prepend the memory latents to the context 
        # rollout n context steps forward
        # evaluate the rollout error
        # report avg rollout error for a trajectory over all rollouts
        
        for idx in range(num_rollouts):
            obs, actions, _, _ = dset[idx]
            obs = {k:v.unsqueeze(0).to(self.device) for k, v in obs.items()}
            actions = actions.unsqueeze(0).to(self.device)

            errors = []
            # loop through trajectory using windows
            # for start_idx in range(0, actions.shape[1], self.window_size):
            # only one window for now
            start_idx = 0
            end_idx = start_idx + self.window_size
            half_hist = self.cfg.num_hist // 2
            obs_w_oracle = {
                k: torch.cat([v[:, half_hist:end_idx], v[:, start_idx:half_hist]], dim=1) for k, v in obs.items()
            }
            act_w_oracle = torch.cat([actions[:, half_hist:end_idx], actions[:, start_idx:half_hist], actions[:, half_hist:end_idx]], dim=1)

            # since model.rollout encodes the context already, we can prepend the oracle to half the history and rollout to end_idx
            z_obses, _ = self.model.rollout(obs_w_oracle, act_w_oracle)
            z_tgts = {k: v[:, :half_hist] for k, v in z_obses.items()} # first half_hist frames are oracle embeddings
            z_obses = {k: v[:, -half_hist:-1] for k, v in z_obses.items()} # last half_hist frames are rollout embeddings

            print(f"z_tgts: {z_tgts['visual'].shape}, z_obses: {z_obses['visual'].shape}")

            decoded_tgt = self.model.decode_obs(z_tgts)[0]
            decoded = self.model.decode_obs(z_obses)[0]

            errors.append({"recon": self.err_eval_single(decoded, decoded_tgt)['visual']})
            errors.append({"pred": self.err_eval_single(z_obses, z_tgts)['visual']})

            # save plots
            if idx < min(10, num_rollouts):
                imgs = torch.cat(
                    [
                        obs_w_oracle["visual"][0, :half_hist].cpu(), # true target
                        decoded_tgt["visual"][0].cpu(), # decoded oracle
                        decoded["visual"][0].cpu(), # decoded pred
                    ],
                    dim=0,
                )
                if self.accelerator.is_main_process:
                    self.plot_imgs(imgs, half_hist, f"{plotting_dir}/e{self.epoch}_{idx}_oracle_{oracle_mode}.png")
                
                





 
        
    

    def save_horizon_results_to_file(self, horizon_logs, filepath):
        if not horizon_logs:
            return
        try:
            keys = sorted(horizon_logs.keys())
            rows = []
            num_rows = len(next(iter(horizon_logs.values())))
            for i in range(num_rows):
                row = {k: horizon_logs[k][i] if i < len(horizon_logs[k]) else None for k in keys}
                rows.append(row)
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            print(f"Error saving horizon results to file: {e}")  

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step

        # Add epoch time to the log message if available
        epoch_time_msg = ""
        if "epoch_time" in epoch_log:
            epoch_time_msg = f"  Epoch time: {epoch_log['epoch_time']:.2f}s"

        # Add learning rates to log message
        lr_msg = f"  Encoder LR: {self.encoder_optimizer.param_groups[0]['lr']:.2e}"
        lr_msg += (
            f"  Predictor LR: {self.predictor_optimizer.param_groups[0]['lr']:.2e}"
        )
        lr_msg += f"  Decoder LR: {self.decoder_optimizer.param_groups[0]['lr']:.2e}"
        lr_msg += f"  Action Encoder LR: {self.action_encoder_optimizer.param_groups[0]['lr']:.2e}"

        log.info(
            f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}{epoch_time_msg}{lr_msg}"
        )

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.cfg.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
            self.plot_imgs(
                imgs,
                num_columns=num_samples * num_frames,
                img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
            )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )

    def get_alpha_values(self):
        """Get current alpha values from the additive control transformer"""
        unwrapped_predictor = self.accelerator.unwrap_model(self.predictor)
        
        if hasattr(unwrapped_predictor, "transformer") and hasattr(
            unwrapped_predictor.transformer, "alphas"
        ):
            if unwrapped_predictor.transformer.alphas is not None:
                module = unwrapped_predictor.transformer
            else:
                return {}
        else:
            return {}

        alphas = {}
        if module.alphas is not None:
            for i, alpha in enumerate(module.alphas):
                alphas[f"alpha_layer_{i}"] = alpha.item()
                if alpha.grad is not None:
                    alphas[f"alpha_layer_{i}_grad_norm"] = alpha.grad.norm().item()
                else:
                    alphas[f"alpha_layer_{i}_grad_norm"] = 0.0
        return alphas


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()
    trainer.accelerator.end_training()
    if trainer.accelerator.is_main_process and hasattr(trainer, "wandb_run"):
        trainer.wandb_run.finish()


if __name__ == "__main__":
    main()
