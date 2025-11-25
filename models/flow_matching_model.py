import math
import torch
from torch._C import parse_schema
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision import transforms
from einops import rearrange, repeat

from models.encoder.resnet import ResNetSmallTokens

class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        image_size,  
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
        K=1,
        normalize_flow=False,
        normalize_target=False,
        integrate_in_loss=False,
        use_pred_loss=True,
        integrator="euler",
        l2_reg_lambda=0.0,
        clamp_tau=1e-6,
        tgt_type="delta",
        use_shortcut_loss=False,
        use_delta_tau=False,
        sigma_min=0.0,
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
        self.K = K
        self.normalize_flow = normalize_flow
        self.normalize_target = normalize_target
        self.integrate_in_loss = integrate_in_loss
        self.use_pred_loss = use_pred_loss
        self.integrator = integrator
        self.l2_reg_lambda = l2_reg_lambda
        self.clamp_tau = clamp_tau
        self.tgt_type = tgt_type
        self.use_shortcut_loss = use_shortcut_loss
        self.use_delta_tau = use_delta_tau
        self.sigma_min = sigma_min

        assert tgt_type == "delta" or tgt_type == "data", f"Invalid tgt type: {tgt_type}"
        if self.use_shortcut_loss:
            assert self.use_delta_tau, "Shortcut loss requires delta tau"

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

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

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

    def normalize(self, z, data_norm=1.0):
        latent = z[..., :-(self.action_dim)]
        latent_norm = torch.norm(latent, dim=-1, keepdim=True) 
        normalized_latent = (latent / latent_norm) * data_norm
        z = torch.cat([normalized_latent, z[..., -(self.action_dim):]], dim=-1)
        return z

    def predict(self, z, tau, delta_tau=None): 
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        if self.use_delta_tau:
            assert delta_tau is not None, "delta_tau must be provided if use_delta_tau is True"
        assert tau is not None, "tau must be provided"

        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        if self.use_delta_tau:
            z, _ = self.predictor(z, tau, delta_tau=delta_tau)
        else:
            z, _ = self.predictor(z, tau)

        z = rearrange(z, "b (t p) d -> b t p d", t=T)

        return z.contiguous()

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

    def sample_tau(self, batch_size, steps):
        steps_log2 = int(torch.log2(torch.tensor(steps))) + 1
        pw2_steps = 2 ** torch.arange(0, steps_log2)

        delta_taus = 1.0 / pw2_steps
        delta_tau = delta_taus[torch.randint(0, len(delta_taus), (batch_size,))]

        taus = (
            torch.floor(
                torch.rand(batch_size, device=delta_tau.device) / delta_tau
            )
            * delta_tau
        )
        return delta_tau, taus

    @torch.no_grad()
    def get_target_flow(self, z_src, z_tgt):
        z_src = z_src.clone()

        if self.tgt_type == "delta":
            target = z_tgt - z_src
        elif self.tgt_type == "data":
            target = z_tgt

        if self.normalize_target:
            target = self.normalize(target)

        if self.use_delta_tau:
            dt, t = self.sample_tau(z_src.size(0), self.K)
            dt = dt.to(z_src.device)
            t = t.view(-1, 1, 1, 1).to(z_src.device)
        else:
            t = torch.rand(z_src.size(0), 1, 1, 1, device=z_src.device)
            dt = torch.full_like(t, 1.0 / self.K, device=z_src.device)

        noise = (
            self.sigma_min # if 0, then simplifies to rectified flows
            * (t * (1 - t))
            * torch.randn_like(z_tgt, device=z_src.device)
        )
        if self.sigma_min > 0:
            target = target + ((1 - 2 * t) * noise)

        z_t = z_src.clone()
        z_t[..., :-self.action_dim] = (1.0 - t) * z_src[..., :-self.action_dim] + (t * z_tgt[..., :-self.action_dim]) + noise[..., : -self.action_dim]
        
        # interpolation
        # z_src_obs, z_src_act = self.separate_emb(z_src)
        # z_tgt_obs, _ = self.separate_emb(z_tgt)

        # z_src_obs["visual"] = (
        #     (1.0 - t) * z_src_obs["visual"]
        #     + (t * z_tgt_obs["visual"])
        #     # + noise[..., : -(self.proprio_dim + self.action_dim)]
        # )
        # # print(f"z_src_obs['proprio'] shape: {z_src_obs['proprio'].shape}")
        # # print(f"noise shape: {noise[..., -(self.proprio_dim + self.action_dim) :-self.action_dim].shape}")

        # z_src_obs["proprio"] = (
        #     (1.0 - t.squeeze(-1)) * z_src_obs["proprio"]
        #     + (t.squeeze(-1) * z_tgt_obs["proprio"])
        #     # + noise[..., -(self.proprio_dim + self.action_dim) :-self.action_dim]
        # )

        # z_t = self.merge_emb(z_src_obs, z_src_act)

        return target.contiguous(), z_t.contiguous(), t.view(t.size(0), 1), dt.view(dt.size(0), 1)

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

        z_norm = torch.norm(z[:, :, :, : -(self.proprio_dim + self.action_dim)], dim=-1).mean()

        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_tgt = obs["visual"][
            :, self.num_pred :, ...
        ]  # (b, num_hist, 3, img_size, img_size)

        # target flow is Enc(x_t) - Enc(x_{t-1})
        target, z_t, t, dt = self.get_target_flow(z_src, z_tgt)

        # flow matching loss
        z_flow = self.predict(z_t, t, delta_tau=dt if self.use_delta_tau else None)
        z_flow_loss = self.emb_criterion(z_flow[:, :, :, : -(self.action_dim)], target[:, :, :, : -(self.action_dim)].detach()) # delta doesnt include action delta

        # shortcut loss
        if self.use_shortcut_loss and self.tgt_type == "delta":
            shortcut_loss = self.shortcut_loss_delta(z_src, target, t, dt, data_norm=z_norm)
            z_flow_loss = z_flow_loss + shortcut_loss
            loss_components["shortcut_loss"] = shortcut_loss
        elif self.use_shortcut_loss and self.tgt_type == "data":
            shortcut_loss = self.shortcut_loss_data(z_src, target, t, dt, data_norm=z_norm)
            z_flow_loss = z_flow_loss + shortcut_loss
            loss_components["shortcut_loss"] = shortcut_loss

        pred_norm = torch.norm(z_flow[:, :, :, : -(self.action_dim)], dim=-1)
        target_norm = torch.norm(target[:, :, :, : -(self.action_dim)].detach(), dim=-1)

        l2_reg_loss = self.l2_reg_lambda * torch.nn.functional.l1_loss(pred_norm, target_norm.detach())
        z_flow_loss = z_flow_loss + l2_reg_loss

        loss_components["pred_norm"] = pred_norm.mean()
        loss_components["target_norm"] = target_norm.mean()
        loss_components["l2_reg_loss"] = l2_reg_loss

        if self.integrate_in_loss and self.tgt_type == "delta":
            z_pred = self.inference(z_src, data_norm=z_norm)
        elif self.tgt_type == "delta":
            z_pred = self.delta_step(z_src, z_flow, torch.ones_like(dt, device=z_src.device), data_norm=z_norm)
        else:
            z_pred = z_flow

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

        loss = loss + z_flow_loss

        if self.use_pred_loss:
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
        loss_components["flow_loss"] = z_flow_loss
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
        act = act.clone()

        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        z_norm = torch.norm(z[:, :, :, : -(self.proprio_dim + self.action_dim)], dim=-1).mean()

        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.inference(z[:, -self.num_hist :], data_norm=z_norm)
            z_t = z_pred[:, -inc:, ...]
            z_t = self.replace_actions_from_z(z_t, action[:, t : t + inc, :])
            z = torch.cat([z, z_t], dim=1)
            t += inc

        z_pred = self.inference(z[:, -self.num_hist :], data_norm=z_norm)
        z_t = z_pred[:, -1:, ...]
        z = torch.cat([z, z_t], dim=1)
        z_obses, _ = self.separate_emb(z)

        return z_obses, z

    def delta_step(self, z, delta, dt, data_norm=1.0):
        if dt.ndim > 1:
            dt = dt.clone().squeeze().view(-1, 1, 1, 1)

        z_new = z.clone()
        z_new[..., :-(self.action_dim)] = z[..., :-(self.action_dim)] + dt * delta[..., :-(self.action_dim)]
        if self.normalize_flow:
            z_new = self.normalize(z_new, data_norm)
        return z_new

    def euler_forward(self, z, K=1, data_norm=1.0):
        dt = 1.0 / K * torch.ones(z.size(0), 1, device=z.device)
        tau = torch.zeros(z.size(0), 1, device=z.device)
        for _ in range(K):
            z_delta = self.predict(z, tau)
            z = self.delta_step(z, z_delta, dt, data_norm=data_norm)
            tau = tau + dt
        return z

    def euler_forward_ckpt(self, z, K=1, data_norm=1.0):
        dt = 1.0 / K * torch.ones(z.size(0), 1, device=z.device)
        tau = torch.zeros(z.size(0), 1, device=z.device)

        def euler_step(z, tau, dt):
            z_pred = self.predict(z, tau, delta_tau=dt)
            if self.tgt_type == "data":
                z_delta = (z_pred - z) / (1.0 - tau.squeeze().view(-1, 1, 1, 1))
            else:
                z_delta = z_pred
            z = self.delta_step(z, z_delta, dt, data_norm=data_norm)
            tau = tau + dt
            return z, tau
        for _ in range(K):
            z, tau = checkpoint.checkpoint(
                euler_step, z, tau, dt, use_reentrant=False
            )

        return z

    def inference(self, z, data_norm=1.0):
        z = z.clone()    
        return self.euler_forward_ckpt(z, K=self.K, data_norm=data_norm)

    def estimate_lipschitz(self, z_pred, z_tgt):
        src = z_pred.clone()
        src_shift = z_pred[:, 1:, ...]
        tgt = z_tgt.clone()
        tgt_shift = z_tgt[:, 1:, ...]

        pred_norm = (src_shift - src[:, :-1, ...]).norm(dim=-1)
        tgt_norm = (tgt_shift - tgt[:, :-1, ...]).norm(dim=-1) + 1e-6
        lipschitz_bound = pred_norm / tgt_norm
        return {
            "mean_bound": torch.mean(lipschitz_bound, dim=1),
            "max_bound": torch.amax(lipschitz_bound, dim=1),
        }

    def shortcut_loss_delta(self, z, target, t, dt, data_norm=1.0):
        target = target.clone()
        b_1 = self.predict(z, t, dt * 0.5) 
        z_prime = self.delta_step(z, b_1, dt * 0.5, data_norm=data_norm)
        b_2 = self.predict(z_prime, t + dt * 0.5, dt * 0.5)
        z_hat = self.predict(z, t, dt)
        shortcut_tgt = (b_1.detach() + b_2.detach()) / 2

        mask = (dt.squeeze() == (1.0 / self.K))
        shortcut_tgt[mask] = target[mask].detach()

        loss = self.emb_criterion(z_hat[..., :-(self.action_dim)], shortcut_tgt[..., :-(self.action_dim)])
        return loss

    def shortcut_loss_data(self, z, target, t, dt, data_norm=1.0):
        target = target.clone()
        b_1 = (self.predict(z, t, dt * 0.5) - z) / (1.0 - t)[:, None, None]
        z_prime = self.delta_step(z, b_1, dt * 0.5, data_norm=data_norm)        
        b_2 = (self.predict(z_prime, t + dt * 0.5, dt * 0.5) - z_prime) / (
            1.0 - (t + dt * 0.5)
        )[:, None, None]

        z_hat = self.predict(z, t, dt)
        shortcut_tgt = (b_1.detach() + b_2.detach()) / 2
        mask = (dt.squeeze() == (1.0 / self.K))
        shortcut_tgt[mask] = target[mask].detach() # dt_min loss: MSE(z_hat, z_tgt)
        z_hat[~mask] = (z_hat[~mask] - z[~mask]) / (1.0 - t[~mask])[:, None, None] # shortcut loss: MSE(z_hat - z, (b1 + b2) / 2)

        loss = torch.nn.functional.mse_loss(z_hat[..., :-(self.action_dim)], shortcut_tgt[..., :-(self.action_dim)], reduction="none")
        loss[~mask] = loss[~mask] * (1.0 - t[~mask][:, None, None])**2
        return loss.mean()
