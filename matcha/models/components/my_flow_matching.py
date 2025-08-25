import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

# Assuming all other necessary imports from your provided code are present
# (e.g., BasicTransformerBlock, ConformerWrapper, SinusoidalPosEmb, etc.)
from abc import ABC
from matcha.models.components.decoder import Decoder # Original for reference
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM_Vec(torch.nn.Module, ABC):
    """
    Modified BASECFM to handle a fixed-size vector output.
    """
    def __init__(
        self,
        n_feats, # This will be the dimension of the condition embeddings (e.g., 768 for BERT)
        output_dim, # The dimension of the target vector (e.g., 256)
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.output_dim = output_dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-4)

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """
        Generates a single vector per item in the batch.

        Args:
            mu (torch.Tensor): Input condition, e.g., text embeddings.
                shape: (batch_size, n_feats, seq_len)
            mask (torch.Tensor): Mask for the input condition.
                shape: (batch_size, 1, seq_len)
            n_timesteps (int): Number of diffusion steps.
        
        Returns:
            sample (torch.Tensor): The generated vector.
                shape: (batch_size, output_dim)
        """
        # z is now a random vector, not a sequence
        z = torch.randn(mu.shape[0], self.output_dim, device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # We need to adapt the solver to handle a vector `x`
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            # The estimator will handle the shape mismatch internally
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """
        Computes diffusion loss for a vector target.

        Args:
            x1 (torch.Tensor): Target vector.
                shape: (batch_size, output_dim)
            mask (torch.Tensor): Mask for the condition `mu`.
                shape: (batch_size, 1, seq_len)
            mu (torch.Tensor): Input condition.
                shape: (batch_size, n_feats, seq_len)
        """
        b, _ = x1.shape

        # random timestep per batch item, shape (b, 1) for broadcasting
        t = torch.rand([b, 1], device=mu.device, dtype=mu.dtype)
        
        # sample noise p(x_0)
        z = torch.randn_like(x1) # z has shape (b, output_dim)

        # Interpolate between noise and target data
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # The estimator's output must match `u`'s shape: (b, output_dim)
        predicted_u = self.estimator(y, mask, mu, t.squeeze(-1), spks)
        
        # Use simple mean squared error; no need for complex masking on the output
        loss = F.mse_loss(predicted_u, u)
        
        return loss, y


class Decoder_Vec(Decoder):
    """
    A modified Decoder (U-Net) that accepts a sequence condition `mu` and a
    noised vector `x`, and outputs a single vector.
    """
    def __init__(
        self,
        in_channels, # This is now mu_dim + x_dim + spk_dim
        out_channels, # This is the final vector dimension (256)
        channels=(256, 256),
        # ... other params are the same as the original Decoder
        **kwargs
    ):
        # We call the original __init__ but will override the final layers.
        super().__init__(in_channels=in_channels, out_channels=out_channels, channels=channels, **kwargs)

        # --- KEY MODIFICATIONS ---
        # 1. The original `final_block` and `final_proj` work on sequences.
        #    We keep the final_block but replace final_proj.
        # self.final_block = Block1D(channels[-1], channels[-1]) # This is already in parent
        
        # 2. Replace the final 1D convolution with a Linear layer.
        #    This layer will project the pooled sequence representation to the final output vector.
        self.upper_proj = nn.Linear(out_channels, channels[-1])
        self.final_proj = nn.Linear(channels[-1], self.out_channels)
        
        # Re-initialize the new layer's weights
        nn.init.kaiming_normal_(self.final_proj.weight, nonlinearity="relu")
        if self.final_proj.bias is not None:
            nn.init.constant_(self.final_proj.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """
        Forward pass for the sequence-to-vector task.

        Args:
            x (torch.Tensor): Noised target vector, shape (batch_size, out_channels)
            mask (torch.Tensor): Mask for mu, shape (batch_size, 1, seq_len)
            mu (torch.Tensor): Condition, shape (batch_size, n_feats, seq_len)
            t (torch.Tensor): Timestep, shape (batch_size,)
        """
        seq_len = mu.shape[-1]
        
        # --- KEY MODIFICATIONS ---

        # 1. Expand the input vector `x` to match the sequence length of `mu`
        # x: (b, out_channels) -> (b, out_channels, seq_len)
        x_expanded = repeat(x, "b c -> b c t", t=seq_len)

        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        # 2. Concatenate the expanded vector, the condition, and speaker embeddings
        # The `in_channels` of the model must be `mu.dim + x.dim + spk.dim`
        to_cat = [x_expanded, mu]
        if spks is not None:
            spks_expanded = repeat(spks, "b c -> b c t", t=seq_len)
            to_cat.append(spks_expanded)
            
        # x_combined: (batch, mu_dim + x_dim + spk_dim, seq_len)
        x_combined = torch.cat(to_cat, dim=1)
        print(f"x_combined: {x_combined.shape}")
        # The rest of the U-Net architecture runs exactly as before
        hiddens = []
        masks = [mask]
        # Down-sampling path
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x_combined = resnet(x_combined, mask_down, t)
            x_combined = rearrange(x_combined, "b c t -> b t c")
            mask_down_flat = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_combined = transformer_block(hidden_states=x_combined, attention_mask=mask_down_flat)
            x_combined = rearrange(x_combined, "b t c -> b c t")
            hiddens.append(x_combined)
            x_combined = downsample(x_combined * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Middle path
        for resnet, transformer_blocks in self.mid_blocks:
            x_combined = resnet(x_combined, mask_mid, t)
            x_combined = rearrange(x_combined, "b c t -> b t c")
            mask_mid_flat = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_combined = transformer_block(hidden_states=x_combined, attention_mask=mask_mid_flat)
            x_combined = rearrange(x_combined, "b t c -> b c t")
        
        # Up-sampling path
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            # Note: pack is not used here as it's an alias for torch.cat with rearrange
            x_combined = torch.cat([x_combined, hiddens.pop()], dim=1)
            x_combined = resnet(x_combined, mask_up, t)
            x_combined = rearrange(x_combined, "b c t -> b t c")
            mask_up_flat = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_combined = transformer_block(hidden_states=x_combined, attention_mask=mask_up_flat)
            x_combined = rearrange(x_combined, "b t c -> b c t")
            x_combined = upsample(x_combined * mask_up)

        # Final block processing on the sequence
        x_seq = self.final_block(x_combined, mask) # Use the original full mask
        
        # 3. Pool the output sequence into a single vector
        # Apply mask before pooling to avoid including padded steps in the mean
        # We add a small epsilon to the sum of the mask to avoid division by zero if a sample is fully masked
        masked_x_seq = x_seq * mask
        pooled_x = torch.sum(masked_x_seq, dim=-1) / (torch.sum(mask, dim=-1) + 1e-8)
        
        # 4. Apply the final Linear projection
        # pooled_x: (b, final_channels) -> output: (b, out_channels)
        output = self.final_proj(pooled_x)
        
        return output


class CFM_Vec(BASECFM_Vec):
    """
    Final model wrapper.
    """
    def __init__(self, text_emb_dim, output_dim, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=text_emb_dim,
            output_dim=output_dim,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        # The input to the U-Net is the concatenation of the expanded target vector,
        # the text embedding condition, and optionally a speaker embedding.
        # output_dim: 256
        # text_emb_dim: 1024
        decoder_in_channels = output_dim + text_emb_dim + (spk_emb_dim if n_spks > 1 else 0)
        
        self.estimator = Decoder_Vec(
            in_channels=decoder_in_channels, 
            out_channels=output_dim, 
            # decoder channels: (256, 512, 1024),
            **decoder_params
        )