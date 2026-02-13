import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

from models.visual_world_model import VWorldModel

class StochasticVWorldModel(VWorldModel):
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
        beta=1.0,  # beta-VAE parameter
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            num_hist=num_hist,
            num_pred=num_pred,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            decoder=decoder,
            predictor=predictor,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            concat_dim=concat_dim,
            num_action_repeat=num_action_repeat,
            num_proprio_repeat=num_proprio_repeat,
            train_encoder=train_encoder,
            train_predictor=train_predictor,
            train_decoder=train_decoder,
            decoder_loss_type=decoder_loss_type,
            step_size=step_size,
            use_cls_token=use_cls_token,
            aux_predictor=aux_predictor,
            per_window_ret_frames=per_window_ret_frames,
            ret_loss_weight=ret_loss_weight,
            max_retention_cache_size=max_retention_cache_size,
            **kwargs,
        )
        
        self.beta = beta
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

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

        if self.predictor is not None:
            z_pred, s = self.predict(z_src)
            if self.decoder is not None:
                # The parent decode method returns (obs, diff) where obs is a dict
                obs_pred, diff_pred = self.decode(z_pred.detach())
                visual_pred = obs_pred["visual"]
                
                recon_loss_pred = self.decoder_criterion(
                    visual_pred, visual_tgt
                )
                
                # For VQ-VAE, we have a commitment loss
                decoder_loss_pred = (
                    recon_loss_pred
                    + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach()
                )
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, : -(self.proprio_dim + self.action_dim)],
                    z_tgt[
                        :, :, :, : -(self.proprio_dim + self.action_dim)
                    ].detach(),
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

            if self.aux_predictor is not None:
                # sample random frame as ctx
                rand_ctx = torch.randint(low=1, high=self.num_hist, size=(1,), dtype=torch.long)
                ret_losses = self.predict_aux(z_src[:, rand_ctx, :, :])

                loss = loss + ret_losses
                loss_components["ret_loss"] = ret_losses

                # add random frames to the retention cache
                rand_idxs = torch.randperm(self.num_hist)[:self.per_window_ret_frames]
                for rand_idx in rand_idxs.tolist():
                    self.add_retention_target(z_src[:, rand_idx, :, :].detach().unsqueeze(1))

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None
            # Initialize loss variables to zero when predictor is None
            z_loss = torch.tensor(0.0, device=z.device)
            z_visual_loss = torch.tensor(0.0, device=z.device)
            z_proprio_loss = torch.tensor(0.0, device=z.device)

        if self.decoder is not None:
            # The parent decode method returns (obs, diff) where obs is a dict
            obs_reconstructed, diff_reconstructed = self.decode(z.detach())
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