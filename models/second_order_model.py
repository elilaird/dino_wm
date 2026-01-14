from sympy.calculus import accumulationbounds
import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

from models.encoder.resnet import ResNetSmallTokens

class SecondOrderModel(nn.Module):
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
        num_action_repeat=0,
        num_proprio_repeat=0,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        decoder_loss_type='mse',
        step_size=1,
        velocity_loss_lambda=0.0,
        kinetic_energy_reg_lambda=0.0,
        rollout_loss_lambda=0.0,
        rollout_k=0,
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
        self.velocity_loss_lambda = velocity_loss_lambda
        self.kinetic_energy_reg_lambda = kinetic_energy_reg_lambda
        self.rollout_loss_lambda = rollout_loss_lambda
        self.rollout_k = rollout_k

        if hasattr(self.predictor, "module"):
            self.dt = self.predictor.module.dt
        else:
            self.dt = self.predictor.dt

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

        if (hasattr(self.encoder, "module") and isinstance(self.encoder.module, ResNetSmallTokens)):
            print(f"Using out_hw from ResNetSmallTokens.module: {self.encoder.module.out_hw}", flush=True)
            num_side_patches = self.encoder.module.out_hw
        elif isinstance(self.encoder, ResNetSmallTokens):
            print(f"Using out_hw from ResNetSmallTokens: {self.encoder.out_hw}", flush=True)
            num_side_patches = self.encoder.out_hw
        else:
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
        act_emb = self.encode_act(act) # b, t, frameskip, action_dim

        proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)

        z = torch.cat(
            [z_dct['visual'], proprio_repeated], dim=3
        )  # (b, num_frames, num_patches, dim + proprio_dim)
        return z, act_emb

    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, frameskip, action_emb_dim)
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

    def predict(self, z, actions):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        # z = rearrange(z, "b t p d -> b (t p) d")

        z, v  = self.predictor(z, actions)
        # z = rearrange(z, "b (t p) d -> b t p d", t=T)
        # v = rearrange(v, "b (t p) d -> b t p d", t=T)
        return z, v

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, _ = self.separate_emb(z)
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
        z_visual, z_proprio = z[..., :-(self.proprio_dim)], \
                                        z[..., -(self.proprio_dim):]
        # remove tiled dimensions
        z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, None

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, frameskip, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}

        z, act_emb = self.encode(obs, act)
        act_emb = act_emb[:, :self.num_hist, ...]
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)

        visual_tgt = obs["visual"][
            :, self.num_pred :, ...
        ]  # (b, num_hist, 3, img_size, img_size)

        z_pred, v_pred  = self.predict(z_src, act_emb)

        if self.rollout_loss_lambda > 0.0 and self.rollout_k > 0:
            rollout_obs = {k: v[:, :1, ...].detach() for k, v in obs.items()}
            _, z_rollout = self.rollout(rollout_obs, act[:, :self.rollout_k, ...].detach())
            rollout_k_loss = self.rollout_loss_lambda * self.emb_criterion(z_rollout[:, -self.rollout_k:, :, :], z_tgt[:, :self.rollout_k, :, :].detach())
            loss = loss + rollout_k_loss
            loss_components["rollout_k_loss"] = rollout_k_loss

        # log log(v_pred_norm)
        v_pred_norm = torch.norm(v_pred, dim=-1)
        loss_components["v_pred_norm"] = torch.log(v_pred_norm + 1e-6).mean()

        if self.kinetic_energy_reg_lambda > 0.0:
            kinetic_energy = 0.5 * v_pred_norm
            kinetic_energy_loss = self.kinetic_energy_reg_lambda * kinetic_energy.mean()
            loss = loss + kinetic_energy_loss
            loss_components["kinetic_energy_reg"] = kinetic_energy_loss

        # velocity magnitude regularization loss
        if self.velocity_loss_lambda > 0.0:
            v_tgt = (z_tgt - z_src) / self.get_dt()
            v_loss = self.velocity_loss_lambda * self.emb_criterion(v_pred, v_tgt.detach())
            loss = loss + v_loss
            loss_components["velocity_loss"] = v_loss

        if self.decoder is not None:
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
        else:
            visual_pred = None

        z_visual_loss = self.emb_criterion(
            z_pred[:, :, :, : -(self.proprio_dim)],
            z_tgt[
                :, :, :, : -(self.proprio_dim)
            ].detach(),
        )
        z_proprio_loss = self.emb_criterion(
            z_pred[
                :,
                :,
                :,
                -(self.proprio_dim) : ,
            ],
            z_tgt[
                :,
                :,
                :,
                -(self.proprio_dim) : ,
            ].detach(),
        )
        z_loss = self.emb_criterion(
            z_pred,
            z_tgt.detach(),
        )

        loss = loss + z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss

        if self.decoder is not None:
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
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    # def replace_actions_from_z(self, z, act):
    #     act_emb = self.encode_act(act)
    #     if self.concat_dim == 0:
    #         z[:, :, -1, :] = act_emb
    #     elif self.concat_dim == 1:
    #         act_tiled = repeat(
    #             act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2]
    #         )
    #         act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
    #         z[..., -self.action_dim :] = act_repeated
    #     return z

    def rollout(self, obs_0, act, bypass_memory_reset=False):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, frameskip, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """

        num_obs_init = obs_0['visual'].shape[1]
        # act_0 = act[:, :num_obs_init]
        # action = act[:, num_obs_init:]
        z, act_emb = self.encode(obs_0, act)
        act_0 = act_emb[:, :num_obs_init]
        action = act_emb[:, num_obs_init:] 

        t = 0
        inc = 1
        while t < act_emb.shape[1] - num_obs_init:
            z_pred, _ = self.predict(z[:, -self.num_hist :], act_emb[:, t: t + inc, ...]) # act: (b, t+inc, frameskip, action_dim) only adding action to last frame
            z_new = z_pred[:, -inc:, ...]
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred, _ = self.predict(z[:, -self.num_hist :], act_emb[:, -1:])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, _ = self.separate_emb(z)

        return z_obses, z

    def set_predictor_step_size(self, step_size):
        pass

    def reset_predictor_memory(self):
        pass

    def set_dt(self, new_dt):
        if hasattr(self.predictor, "set_dt"):
            self.predictor.set_dt(new_dt)
            assert self.predictor.dt == new_dt, f"Predictor dt {self.predictor.dt} does not match new_dt {new_dt}"
            # print(f"Set predictor dt to {new_dt}")
        elif hasattr(self.predictor, "module") and hasattr(self.predictor.module, "set_dt"):
            self.predictor.module.set_dt(new_dt)
            assert self.predictor.module.dt == new_dt, f"Predictor dt {self.predictor.module.dt} does not match new_dt {new_dt}"
            # print(f"Set predictor dt to {new_dt}")
        else:
            raise ValueError(f"Predictor {self.predictor} does not have a set_dt method")

    def get_dt(self):
        if hasattr(self.predictor, "get_dt"):
            return self.predictor.get_dt()
        elif hasattr(self.predictor, "module") and hasattr(self.predictor.module, "get_dt"):
            return self.predictor.module.get_dt()
        else:
            raise ValueError(f"Predictor {self.predictor} does not have a get_dt method")
