import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BetaVAEDecoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=256,
        beta=1.0,  # beta-VAE parameter for KL divergence weighting
        latent_dim=128,  # dimension of the latent space
    ):
        super().__init__()
        
        self.beta = beta
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        
        # Encoder to get latent distribution parameters
        self.encoder_fc = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and logvar
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        
        # Upsampling to image space (simplified version)
        self.upsample = nn.Sequential(
            nn.Conv2d(emb_dim, channel, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channel // 4, in_channel, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.info = f"in_channel: {in_channel}, channel: {channel}, emb_dim: {emb_dim}, beta: {beta}"

    def encode(self, input):
        """
        Encode input to latent distribution parameters
        input: (b, t, num_patches, emb_dim)
        """
        batch_size, t, num_patches, emb_dim = input.shape
        input_flat = input.contiguous().view(-1, emb_dim)  # (b*t*num_patches, emb_dim)
        h = self.encoder_fc(input_flat)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_to_image(self, emb):
        """
        Decode embeddings to images
        emb: (b, t, num_patches, emb_dim)
        """
        b, t, num_patches, emb_dim = emb.shape
        num_side_patches = int(num_patches ** 0.5)
        
        # Reshape embeddings to spatial format
        emb_reshaped = rearrange(emb, "b t (h w) e -> (b t) e h w", h=num_side_patches, w=num_side_patches)
        
        # Upsample to image
        decoded = self.upsample(emb_reshaped)  # (b*t, 3, 64, 64)
        decoded = rearrange(decoded, "(b t) c h w -> b t c h w", b=b, t=t)
        return decoded

    def forward(self, input):
        """
        input: (b, t, num_patches, emb_dim)
        """
        # Encode
        mu, logvar = self.encode(input)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode to embeddings
        recon_emb_flat = self.decoder_fc(z)  # (b*t*num_patches, emb_dim)
        
        # Reshape reconstructed embeddings
        b, t, num_patches, emb_dim = input.shape
        recon_emb = recon_emb_flat.view(b, t, num_patches, emb_dim)
        
        # Decode to images
        recon_img = self.decode_to_image(recon_emb)
        
        # Calculate KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Return reconstructed images and KL divergence
        return recon_img, kl_div

    def sample(self, num_samples, device):
        """
        Generate samples from the model
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder_fc(z)
        return samples