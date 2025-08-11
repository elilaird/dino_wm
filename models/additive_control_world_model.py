import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from models.visual_world_model import VWorldModel

class AdditiveControlVWorldModel(VWorldModel):
    """
    VWorldModel with additive control injection for actions.
    Separates action processing from visual/proprio processing to avoid gradient entanglement.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized AdditiveControlVWorldModel with additive control injection")
    
    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim) - visual/proprio only
                   act_emb (tensor): (b, num_frames, action_emb_dim) - separate action embeddings
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        
        # For additive control, we return visual/proprio embeddings separately from actions
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2 # visual + proprio only
                )  # (b, num_frames, num_patches + 1, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + proprio_dim)
        return z, act_emb
    
    def predict(self, z, actions=None):  # Modified to support additive control
        """
        input : z: (b, num_hist, num_patches, emb_dim) - visual/proprio embeddings only
                actions: (b, num_hist, action_dim) - separate action input (optional)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        
        # Check if predictor supports additive control
        if hasattr(self.predictor, 'action_dim') and self.predictor.action_dim > 0:
            # Use additive control if actions provided and predictor supports it
            if actions is not None:
                z = self.predictor(z, actions)
            else:
                z = self.predictor(z)
        else:
            # Fallback to standard prediction
            z = self.predictor(z)
            
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z
    
    def separate_emb(self, z):
        """
        input: z (tensor) - visual/proprio embeddings only (no actions)
        output: z_obs (dict), z_act (tensor) - dummy action tensor for compatibility
        """
        if self.concat_dim == 0:
            z_visual, z_proprio = z[:, :, :-1, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio = z[..., :-(self.proprio_dim)], z[..., -(self.proprio_dim):]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        # Return dummy action tensor for compatibility with existing code
        z_act = torch.zeros(z.shape[0], z.shape[1], self.action_dim, device=z.device)
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z, act_emb = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        act_src = act_emb[:, : self.num_hist, :]  # (b, num_hist, action_emb_dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            # Pass actions to predictor for additive control
            z_pred = self.predict(z_src, act_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -1, :], z_tgt[:, :, -1, :].detach())
                z_loss = self.emb_criterion(z_pred, z_tgt.detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim):], 
                    z_tgt[:, :, :, -(self.proprio_dim):].detach()
                )
                z_loss = self.emb_criterion(z_pred, z_tgt.detach())

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            # For reconstruction, we need to concatenate visual/proprio with actions
            # to maintain compatibility with existing decoder
            if self.concat_dim == 0:
                z_recon = torch.cat([z, act_emb.unsqueeze(2)], dim=2)
            elif self.concat_dim == 1:
                act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
                act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
                z_recon = torch.cat([z, act_repeated], dim=3)
            
            obs_reconstructed, diff_reconstructed = self.decode(
                z_recon.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def rollout(self, obs_0, act):
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
        z, _ = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            # Pass actions to predictor for additive control
            z_pred = self.predict(z[:, -self.num_hist :], action[:, t : t + inc, :])
            z_new = z_pred[:, -inc:, ...]
            z = torch.cat([z, z_new], dim=1)
            t += inc

        # Final prediction
        z_pred = self.predict(z[:, -self.num_hist :], action[:, -1:, :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
