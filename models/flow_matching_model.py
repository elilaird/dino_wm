import math
import torch
from torch._C import parse_schema
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

from models.encoder.resnet import ResNetSmallTokens

class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        decoder_loss_type='mse',
        step_size=1,
        use_cls_token=False,
        aux_predictor=None,
        per_window_ret_frames=2, # number of frames to cache for retention
        ret_loss_weight=1.0,
        max_retention_cache_size=10,
        input_type="causal",
        interpolation_type="linear",
        sigma0=0.1,
        K=1,
        **kwargs,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat
        self.decoder_loss_type = decoder_loss_type 
        self.step_size = step_size
        self.use_cls_token = use_cls_token
        self.aux_predictor = aux_predictor
        self.per_window_ret_frames = per_window_ret_frames
        self.ret_loss_weight = ret_loss_weight
        self.max_retention_cache_size = max_retention_cache_size
        self.input_type = input_type
        self.interpolation_type = interpolation_type
        self.sigma0 = sigma0
        self.K = K

        if hasattr(self.encoder, "module"):
            self.emb_dim = self.encoder.module.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
            encoder_patch_size = self.encoder.module.patch_size
        else:
            self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
            encoder_patch_size = self.encoder.patch_size

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        decoder_scale = 16  # from vqvae
        print(f"Using decoder_scale from cfg: {image_size // decoder_scale}", flush=True)
        num_side_patches = image_size // decoder_scale            

        self.encoder_image_size = num_side_patches * encoder_patch_size
        print(f"Encoder image size: {self.encoder_image_size}", flush=True)
        self.encoder_transform = transforms.Compose(
            [transforms.Resize(self.encoder_image_size)]
        )

        # Initialize decoder criterion based on config
        self.decoder_loss_type = getattr(self, 'decoder_loss_type', 'mse')
        if self.decoder_loss_type == 'smooth_l1':
            self.decoder_criterion = nn.SmoothL1Loss()
        else:  # default to mse
            self.decoder_criterion = nn.MSELoss()
        print(f"Decoder loss type: {self.decoder_criterion}")

        if self.aux_predictor is not None:
            self.aux_predictor_criterion = nn.MSELoss()

        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)
        if self.aux_predictor is not None:
            self.aux_predictor.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.aux_predictor is not None:
            self.aux_predictor.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z

    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z, tau=None):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        if tau is not None:
            z, _ = self.predictor(z, tau)
        else:
            z, _ = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)

        return z

    def predict_aux(self, ctx):
        pass

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        z = {k: v.clone() for k, v in z_obs.items()}
        if self.use_cls_token:
            # remove cls token
            z["visual"] = z["visual"][:, :, 1:, :]
        visual, diff = self.decoder(z["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff

    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_visual = z_visual.contiguous()
        z_proprio = z_proprio.contiguous()
        z_act = z_act.contiguous()
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def merge_emb(self, z_obs, z_act):
        if self.concat_dim == 0:
            z = torch.cat(
                [
                    z_obs["visual"],
                    z_obs["proprio"].unsqueeze(2),
                    z_act.unsqueeze(2),
                ],
                dim=2,  # add as an extra token
            )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(
                z_obs["proprio"].unsqueeze(2),
                "b t 1 a -> b t f a",
                f=z_obs["visual"].shape[2],
            )
            proprio_repeated = proprio_tiled.repeat(
                1, 1, 1, self.num_proprio_repeat
            )
            act_tiled = repeat(
                z_act.unsqueeze(2),
                "b t 1 a -> b t f a",
                f=z_obs["visual"].shape[2],
            )
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_obs["visual"], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z

    def sine_sigma(self, t, sigma0=0.2):
        sigma = sigma0 * torch.sin(math.pi * t)
        sigmap = sigma0 * math.pi * torch.cos(math.pi * t)
        return sigma, sigmap

    # @torch.no_grad()
    # def get_target_flow(self, z_src, z_tgt):
    #     z_src = z_src.clone()

    #     t = torch.rand(z_src.size(0), 1, 1, 1, device=z_src.device)
    #     z_src_obs, z_src_act = self.separate_emb(z_src)
    #     z_tgt_obs, _ = self.separate_emb(z_tgt)

    #     z_src_obs_original = {k: v.clone() for k, v in z_src_obs.items()}

    #     z_src_obs["visual"] = (1.0 - t) * z_src_obs["visual"] + (t * z_tgt_obs["visual"])
    #     z_src_obs["proprio"] = (1.0 - t.squeeze(-1)) * z_src_obs["proprio"] + (t.squeeze(-1) * z_tgt_obs["proprio"])

    #     if self.interpolation_type == "nonlinear":
    #         sigma_t, sigmap_t = self.sine_sigma(t, sigma0=self.sigma0)
    #         eps_vis = torch.randn_like(z_src_obs["visual"]) * sigma_t
    #         eps_proprio = torch.randn_like(z_src_obs["proprio"]) * sigma_t.squeeze(-1)

    #         # mu_0 + eps_t
    #         z_src_obs["visual"] = z_src_obs["visual"] + eps_vis
    #         z_src_obs["proprio"] = z_src_obs["proprio"] + eps_proprio

    #         # target: (z_1 - z_0) + (sigma'(t)/sigma(t)) * (z_t - mu_t)
    #         ratio = sigmap_t / torch.clamp(sigma_t, min=1e-6)
    #         delta_vis = z_tgt_obs['visual'] - z_src_obs_original['visual'] + ratio * (z_tgt_obs['visual'] - z_src_obs['visual'])
    #         delta_proprio = z_tgt_obs['proprio'] - z_src_obs_original['proprio'] + ratio.squeeze(-1) * (z_tgt_obs['proprio'] - z_src_obs['proprio'])

    #     else:
    #         delta_vis = z_tgt_obs['visual'] - z_src_obs_original['visual']
    #         delta_proprio = z_tgt_obs['proprio'] - z_src_obs_original['proprio']

    #     delta_vis = delta_vis.contiguous()
    #     delta_proprio = delta_proprio.contiguous()
    #     z_src_obs['visual'] = z_src_obs['visual'].contiguous()
    #     z_src_obs['proprio'] = z_src_obs['proprio'].contiguous()
    #     z_src_act = z_src_act.contiguous()

    #     delta = self.merge_emb({"visual": delta_vis, "proprio": delta_proprio}, torch.zeros_like(z_src_act, device=z_src.device))

    #     # delta = z_tgt - z_src
    #     z_t = self.merge_emb(z_src_obs, z_src_act)

    #     return delta, z_t
    @torch.no_grad()
    def get_target_flow(self, z_src, z_tgt):
        z_src = z_src.clone()
        delta = z_tgt - z_src

        if self.input_type == "causal":
            z_t = z_src

        # interpolation
        elif self.input_type == "interp":
            t = torch.rand(z_src.size(0), 1, 1, 1, device=z_src.device)

            z_src_obs, z_src_act = self.separate_emb(z_src)
            z_tgt_obs, _ = self.separate_emb(z_tgt)

            z_src_obs["visual"] = (1.0 - t) * z_src_obs["visual"] + (
                t * z_tgt_obs["visual"]
            )
            z_src_obs["proprio"] = (1.0 - t.squeeze(-1)) * z_src_obs["proprio"] + (t.squeeze(-1) * z_tgt_obs["proprio"])

            z_t = self.merge_emb(z_src_obs, z_src_act)
        else:
            raise ValueError(f"Invalid input type: {self.input_type}")

        return delta, z_t

    def add_timestep_embedding(self, z, tau):
        pass

    def forward(self, obs, act, aux_obs=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                aux_obs: "visual", "proprio", "actions"
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}

        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs["visual"][
            :, : self.num_hist, ...
        ]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs["visual"][
            :, self.num_pred :, ...
        ]  # (b, num_hist, 3, img_size, img_size)

        # case 1: target flow is Enc(x_t) - Enc(x_{t-1})
        delta, z_t = self.get_target_flow(z_src, z_tgt)

        z_flow = self.predict(z_t)
        loss = loss +  self.emb_criterion(z_flow[:, :, :, : -(self.action_dim)], delta[:, :, :, : -(self.action_dim)].detach()) # delta doesnt include action delta

        z_pred = z_src + z_flow

        z_visual_loss = self.emb_criterion(
            z_pred[:, :, :, : -(self.proprio_dim + self.action_dim)],
            z_tgt[:, :, :, : -(self.proprio_dim + self.action_dim)].detach(),
        )
        z_proprio_loss = self.emb_criterion(
                    z_pred[
                        :,
                        :,
                        :,
                        -(
                            self.proprio_dim + self.action_dim
                        ) : -self.action_dim,
                    ],
                    z_tgt[
                        :,
                        :,
                        :,
                        -(
                            self.proprio_dim + self.action_dim
                        ) : -self.action_dim,
                    ].detach(),
                )
        z_loss = self.emb_criterion(
            z_pred[:, :, :, : -self.action_dim],
            z_tgt[:, :, :, : -self.action_dim].detach(),
        )
        loss = loss + z_visual_loss + z_proprio_loss + z_loss

        if self.decoder is not None:
            # decoding from z_pred (not connected to predictor loss)
            obs_pred, diff_pred = self.decode(
                z_pred.detach()
            )  # recon loss should only affect decoder
            visual_pred = obs_pred["visual"]
            recon_loss_pred = self.decoder_criterion(
                visual_pred, visual_tgt
            )
            decoder_loss_pred = (
                recon_loss_pred
                + self.decoder_latent_loss_weight * diff_pred
            )
            loss_components["decoder_recon_loss_pred"] = recon_loss_pred
            loss_components["decoder_vq_loss_pred"] = diff_pred
            loss_components["decoder_loss_pred"] = decoder_loss_pred

            # reconstruction loss only affects decoder
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(
                visual_reconstructed, obs["visual"]
            )
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = (
                diff_reconstructed
            )
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_pred = None
            visual_reconstructed = None

        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss
        loss_components["z_loss"] = z_loss
        loss_components["flow_loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z

    def rollout(self, obs_0, act, bypass_memory_reset=False):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """

        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)

        t = 0
        inc = 1
        z_t = z[:,-1:, ...]
        while t < action.shape[1]:
            z_delta = self.inference(z[:, -self.num_hist :])
            # z_t[:, -inc:, :, : -(self.action_dim)] = z_t[:, -inc:, :, : -(self.action_dim)] + z_delta[:, -inc:, :, : -(self.action_dim)] # don't add action delta to z_t to prevent prev action corruption
            z_t = z_t + z_delta[:, -inc:, ...]
            z_t = self.replace_actions_from_z(z_t, action[:, t : t + inc, :])
            z = torch.cat([z, z_t], dim=1)
            t += inc

        z_delta = self.inference(z[:, -self.num_hist :])
        # z_t[:, -1:, :, : -(self.action_dim)] = z_t[:, -1:, :, : -(self.action_dim)] + z_delta[:, -1 :, :, : -(self.action_dim)]
        z_t = z_t + z_delta[:, -1 :, ...]
        z = torch.cat([z, z_t], dim=1)
        z_obses, _ = self.separate_emb(z)

        return z_obses, z

    def euler_forward(self, z_src, K=1):
        z = z_src.clone()
        h = 1.0 / K
        tau = torch.zeros(z_src.size(0), 1, device=z_src.device)
        for _ in range(K):
            z_delta = self.predict(z, tau)
            z[:,-1:,:, :-(self.action_dim)] = z[:,-1:,:, :-(self.action_dim)] + h * z_delta[:,-1:,:, :-(self.action_dim)]
            tau = tau + h
        return z
    
    def inference(self, z):
        if self.input_type == "causal":
            return self.predict(z)
        elif self.input_type == "interp":
            return self.euler_forward(z, K=self.K)
        else:
            raise ValueError(f"Invalid inference type: {self.input_type}")