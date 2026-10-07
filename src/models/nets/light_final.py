"""DynaConTalk denoiser.

x0-prediction network on wavelet-domain motion windows:
  motion encoder   body: per-joint, per-band tokens -> spatial Transformer (joints) and
                   cross-band Transformer, fused per frame, plus a linear bypass;
                   face: a linear projection
  trunk            stages of local convolutions and a patch-level temporal Transformer,
                   each stage conditioned by the speech conditioning network (FiLM and a
                   gated feature injection)
  decoder          one head per wavelet band
"""
import torch
import torch.nn as nn
from einops import repeat

from src.models.nets.audio_conditioning import AudioConditioning
from src.models.utils.embedding import PositionEmbedding, timestep_embedding

POSE_DIM = 330  # 55 SMPL-X joints x 6D rotation


class WaveletBandHeads(nn.Module):
    """Independent decoder head for each interleaved wavelet band."""

    def __init__(self, input_dim: int, base_channels: int, num_bands: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.base_channels = int(base_channels)
        self.num_bands = int(num_bands)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, self.base_channels),
                )
                for _ in range(self.num_bands)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = [head(x) for head in self.heads]
        return torch.stack(bands, dim=-1).reshape(*x.shape[:-1], self.base_channels * self.num_bands)


class LightT2M(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        audio_conditioning_cfg: dict,
        input_motion_dim: int = None,
        max_motion_len: int = 64,
        stage_dim: str = "1024*4",
        num_groups: int = 16,
        patch_size: int = 8,
        target_group: str = "body",
        wavelet_levels: int = 3,
        num_joints: int = 55,
        use_joint_tokens: bool = False,
        joint_embed_dim: int = 64,
        joint_transformer_layers: int = 1,
        joint_transformer_heads: int = 4,
        joint_transformer_ffn_dim: int = None,
        joint_dropout: float = 0.0,
        cross_band_transformer_layers: int = 2,
        cross_band_transformer_heads: int = 4,
        cross_band_transformer_ffn_dim: int = None,
        cross_band_batch_chunk: int = 16384,
        use_encoder_bypass: bool = False,
        band_decoder_hidden: int = 2048,
        decoder_dropout: float = 0.0,
        temporal_transformer_heads: int = 8,
        temporal_transformer_ffn_ratio: float = 2.0,
        temporal_transformer_dropout: float = 0.0,
        proposal_scale_init: float = 1.0,
        use_identity_token: bool = False,
    ):
        """
        motion_dim             wavelet motion size: body 1320 (330 x 4 bands), face 400 (100 x 4)
        audio_conditioning_cfg arguments of AudioConditioning
        stage_dim              trunk widths, "<dim>*<stages>"
        target_group           "body" or "face"
        use_joint_tokens       body only: per-joint / per-band token encoder (else a linear projection)
        use_encoder_bypass     add a linear projection of the input to the token encoder output,
                               with the token branch scaled by a zero-initialized gain
        use_identity_token     add the projected speaker identity to the time token
        """
        super().__init__()
        if target_group not in {"body", "face"}:
            raise ValueError(f"target_group must be 'body' or 'face', got {target_group!r}")
        base_dim = int(stage_dim.split("*")[0])
        stage_dims = [base_dim] * int(stage_dim.split("*")[1])
        self.pos_emb = PositionEmbedding(max_motion_len + 1, base_dim, dropout=0.1)  # +1: time token

        self.motion_dim = motion_dim
        self.input_motion_dim = input_motion_dim if input_motion_dim is not None else motion_dim
        self.target_group = target_group
        self.use_joint_tokens = use_joint_tokens
        self.num_bands = wavelet_levels + 1
        self.channel_dim = motion_dim // self.num_bands  # 330 for body, 100 for face
        if target_group == "body" and self.channel_dim != POSE_DIM:
            raise ValueError(f"body target expects {POSE_DIM} channels per band, got {self.channel_dim}")
        if use_joint_tokens and target_group != "body":
            raise ValueError("use_joint_tokens requires target_group='body'")
        self.num_joints = num_joints
        self.joint_embed_dim = joint_embed_dim
        self.cross_band_batch_chunk = int(cross_band_batch_chunk)
        self.base_dim = base_dim
        self.stage_dims = stage_dims
        self.patch_size = patch_size

        if not use_joint_tokens:
            self.motion_proj = nn.Linear(self.input_motion_dim, base_dim)
        self.time_emb = nn.Linear(base_dim, base_dim)

        self.conditioning_module = AudioConditioning(
            stage_dims=stage_dims, patch_size=patch_size, base_dim=base_dim, **audio_conditioning_cfg
        )
        self._condition_cache = None

        # Speaker identity token: the conditioning network's identity embedding (speaker id +
        # body shape), projected with zero initialization and added to the time token, which
        # every trunk layer attends to. It is kept in both CFG branches, so guidance acts on
        # the speech content only.
        self.identity_token_proj = None
        if use_identity_token:
            self.identity_token_proj = nn.Linear(audio_conditioning_cfg["embed_dim"], base_dim)
            nn.init.zeros_(self.identity_token_proj.weight)
            nn.init.zeros_(self.identity_token_proj.bias)

        self.layers = nn.ModuleList(
            [
                StageBlock(
                    dim,
                    num_groups=num_groups,
                    patch_size=patch_size,
                    transformer_heads=temporal_transformer_heads,
                    transformer_ffn_ratio=temporal_transformer_ffn_ratio,
                    transformer_dropout=temporal_transformer_dropout,
                    proposal_scale_init=proposal_scale_init,
                )
                for dim in stage_dims
            ]
        )

        if use_joint_tokens:
            rotation_dim = POSE_DIM // num_joints
            self.body_band_convs = nn.ModuleList(
                [nn.Conv1d(rotation_dim, joint_embed_dim, kernel_size=3, padding=1) for _ in range(self.num_bands)]
            )
            self.joint_identity_embedding = nn.Parameter(torch.empty(1, 1, num_joints, 1, joint_embed_dim))
            self.band_identity_embedding = nn.Parameter(torch.empty(1, 1, 1, self.num_bands, joint_embed_dim))
            nn.init.normal_(self.joint_identity_embedding, std=0.02)
            nn.init.normal_(self.band_identity_embedding, std=0.02)
            self.spatial_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=joint_embed_dim,
                    nhead=joint_transformer_heads,
                    dim_feedforward=joint_transformer_ffn_dim or joint_embed_dim * 2,
                    dropout=joint_dropout,
                    batch_first=True,
                ),
                num_layers=joint_transformer_layers,
            )
            self.cross_band_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=joint_embed_dim,
                    nhead=cross_band_transformer_heads,
                    dim_feedforward=cross_band_transformer_ffn_dim or joint_embed_dim * 4,
                    dropout=joint_dropout,
                    batch_first=True,
                ),
                num_layers=cross_band_transformer_layers,
            )
            self.band_fusion = nn.Sequential(
                nn.Linear(self.num_bands * joint_embed_dim, joint_embed_dim),
                nn.LayerNorm(joint_embed_dim),
                nn.GELU(),
            )
            self.joint_fusion = nn.Linear(num_joints * joint_embed_dim, base_dim)
            self.joint_token_bypass = None
            if use_encoder_bypass:
                self.joint_token_bypass = nn.Linear(self.input_motion_dim, base_dim)
                self.joint_token_norm = nn.LayerNorm(base_dim)
                self.joint_token_scale = nn.Parameter(torch.zeros(1))

        decoder = WaveletBandHeads(base_dim, self.channel_dim, self.num_bands, int(band_decoder_hidden), decoder_dropout)
        if target_group == "body":
            self.band_decoder_body = decoder
        else:
            self.band_decoder_face = decoder

    def _encode_joint_tokens(self, motion: torch.Tensor) -> torch.Tensor:
        """[B, T, 1320] interleaved wavelet motion -> [B, T, base_dim]."""
        b, t, _ = motion.shape
        l1 = self.num_bands
        rotation_dim = POSE_DIM // self.num_joints
        x = motion.view(b, t, self.channel_dim, l1)
        rot = x.view(b, t, self.num_joints, rotation_dim, l1).permute(0, 1, 2, 4, 3)  # [B, T, J, L1, 6]

        band_tokens = []
        for band_idx, band_conv in enumerate(self.body_band_convs):
            band = rot[:, :, :, band_idx, :].permute(0, 2, 3, 1).reshape(b * self.num_joints, rotation_dim, t)
            band = band_conv(band).view(b, self.num_joints, self.joint_embed_dim, t).permute(0, 3, 1, 2)
            band_tokens.append(band)
        tokens = torch.stack(band_tokens, dim=3)  # [B, T, J, L1, E]
        tokens = tokens + self.joint_identity_embedding + self.band_identity_embedding

        # The same spatial Transformer is shared across bands.
        spatial = tokens.permute(0, 1, 3, 2, 4).reshape(b * t * l1, self.num_joints, self.joint_embed_dim)
        spatial = self.spatial_transformer(spatial, mask=None)
        tokens = spatial.view(b, t, l1, self.num_joints, self.joint_embed_dim).permute(0, 1, 3, 2, 4)

        cross_band = tokens.reshape(b * t * self.num_joints, l1, self.joint_embed_dim)
        # Large batches create more independent 4-token sequences than the CUDA attention
        # launch grid allows; chunking is exact because the sequences do not interact.
        if cross_band.shape[0] > self.cross_band_batch_chunk:
            cross_band = torch.cat(
                [self.cross_band_transformer(chunk) for chunk in cross_band.split(self.cross_band_batch_chunk, dim=0)],
                dim=0,
            )
        else:
            cross_band = self.cross_band_transformer(cross_band)
        cross_band = cross_band.view(b, t, self.num_joints, l1 * self.joint_embed_dim)
        joint_tokens = self.band_fusion(cross_band).reshape(b, t, self.num_joints * self.joint_embed_dim)
        return self.joint_fusion(joint_tokens)

    def forward(self, motion, motion_mask, timestep, speech, activity, cond_drop_mask=None, **conditions):
        """
        motion          [B, T, D] noisy motion window (history + future)
        motion_mask     [B, T] bool, valid frames
        timestep        [B] diffusion timesteps
        speech          {"rhythm", "semantic", "mel"} per-frame speech features
        activity        [B, activity_dim] speaker id
        cond_drop_mask  [B] bool, samples using the null condition (classifier-free guidance)
        conditions      clip_text, shape_betas, trajectory_cond, keypose_hint, keypose_mask
        Returns the predicted clean window [B, T, D].
        """
        if self.use_joint_tokens:
            motion_encoded = self._encode_joint_tokens(motion)
            if self.joint_token_bypass is not None:
                motion_encoded = self.joint_token_bypass(motion) + self.joint_token_scale * self.joint_token_norm(motion_encoded)
        else:
            motion_encoded = self.motion_proj(motion)

        time_emb = self.time_emb(timestep_embedding(timestep, motion_encoded.shape[-1])).unsqueeze(dim=1)
        time_mask = torch.ones([time_emb.shape[0], 1], dtype=torch.bool, device=time_emb.device)
        x = torch.cat([time_emb, motion_encoded], dim=1)
        x_mask = torch.cat([time_mask, motion_mask], dim=1)

        condition = self.conditioning_module(
            speech,
            x_mask,
            activity=activity,
            cond_drop_mask=cond_drop_mask,
            motion_context=motion_encoded,
            timestep_context=time_emb[:, 0],
            timestep_values=timestep,
            **conditions,
        )
        self._condition_cache = condition

        # DGN v2 frame-level rhythm term, added to the motion tokens (not the time token)
        frame_conditioning = condition.get("frame_conditioning")
        if frame_conditioning is not None:
            seq_len = x.shape[1] - 1
            fc = frame_conditioning
            if fc.shape[1] < seq_len:
                fc = torch.cat([fc, fc[:, -1:].expand(-1, seq_len - fc.shape[1], -1)], dim=1)
            x = torch.cat([x[:, :1], x[:, 1:] + fc[:, :seq_len]], dim=1)

        if self.identity_token_proj is not None:
            tok = self.identity_token_proj(condition["identity_global"].to(dtype=x.dtype)).unsqueeze(1)
            x = torch.cat([x[:, :1] + tok, x[:, 1:]], dim=1)

        x = self.pos_emb(x)
        for layer, stage_cond in zip(self.layers, condition["stage_conditions"]):
            layer.mixed_module.set_conditioning(stage_cond["feature"], stage_cond["film_gamma"], stage_cond["film_beta"])
            x = layer(x, x_mask)

        out = x[:, 1:]  # drop the time token
        if self.target_group == "body":
            return self.band_decoder_body(out)
        return self.band_decoder_face(out)

    def pop_condition_cache(self):
        cache = self._condition_cache
        self._condition_cache = None
        return cache


class LocalModule(nn.Module):
    """Pointwise + depthwise temporal convolution with a residual LayerNorm."""

    def __init__(self, model_dim, num_groups=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.Conv1d(model_dim, model_dim, 3, 1, 1, groups=model_dim),
            nn.GroupNorm(num_groups=num_groups, num_channels=model_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, x_mask):
        x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask], device=x.device)
        return self.norm(x + self.conv(x.permute([0, 2, 1])).permute([0, 2, 1]))


class TemporalTransformerMixer(nn.Module):
    """Pre-LN Transformer block over the patch sequence."""

    def __init__(self, model_dim, num_heads=8, ffn_ratio=2.0, dropout=0.0):
        super().__init__()
        ffn_dim = max(model_dim, int(model_dim * ffn_ratio))
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + x_attn
        return x + self.ffn(self.norm2(x))


class MixedModule(nn.Module):
    """Patch-level stage: downsample, condition injection, temporal Transformer, upsample, fuse."""

    def __init__(self, model_dim, patch_size=8, transformer_heads=8, transformer_ffn_ratio=2.0,
                 transformer_dropout=0.0, proposal_scale_init=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.local_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, patch_size, patch_size, 0, groups=model_dim),
        )
        self.global_transformer = TemporalTransformerMixer(
            model_dim, num_heads=transformer_heads, ffn_ratio=transformer_ffn_ratio, dropout=transformer_dropout
        )
        self.final_fc = nn.Linear(model_dim * 2, model_dim)
        self.norm = nn.LayerNorm(model_dim)

        # condition injection: a proposal p from (motion, condition), a gate g from (p, motion),
        # update = alpha * tanh(g) * p; the gate network's last layer starts at zero (no injection)
        self.physics_feature_predictor = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )
        self.feature_consensus_network = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim // 2),
            nn.Tanh(),
            nn.Linear(model_dim // 2, model_dim),
        )
        self.injection_scale = nn.Parameter(torch.tensor(float(proposal_scale_init)))
        nn.init.zeros_(self.feature_consensus_network[-1].weight)
        nn.init.zeros_(self.feature_consensus_network[-1].bias)

    def set_conditioning(self, feature, film_gamma, film_beta):
        self._cond_feature = feature
        self._film_gamma = film_gamma
        self._film_beta = film_beta

    def inject_condition(self, x, feature, film_gamma, film_beta):
        x = x * (1.0 + torch.tanh(film_gamma)) + torch.tanh(film_beta)
        proposal = self.physics_feature_predictor(torch.cat([x, feature], dim=-1))
        gate = torch.tanh(self.feature_consensus_network(torch.cat([proposal, x], dim=-1)))
        return x + self.injection_scale * (gate * proposal)

    def forward(self, x, x_mask):
        x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask], device=x.device)
        B, L, D = x.shape
        x1 = x[:, 1:]  # motion frames (without the time token)
        padding_size = x1.shape[1] % self.patch_size
        if padding_size != 0:
            x1 = torch.cat([x1, torch.zeros([B, self.patch_size - padding_size, D], device=x1.device)], dim=1)

        patches = self.local_conv(x1.permute([0, 2, 1])).permute([0, 2, 1])  # [B, T / patch, D]
        x2 = self.inject_condition(patches, self._cond_feature, self._film_gamma, self._film_beta)

        # bidirectional context: the reversed sequence is prepended and the second half kept
        x2 = self.global_transformer(torch.cat([x2.flip([1]), x2], dim=1))[:, x2.shape[1]:]
        x2 = repeat(x2, "B L D -> B (L S) D", S=self.patch_size)

        original_length = x1.shape[1] - padding_size if padding_size != 0 else x1.shape[1]
        x2 = x2[:, :original_length, :]
        x1 = x1[:, :original_length]
        out = torch.cat([x[:, :1], self.final_fc(torch.cat([x1, x2], dim=-1))], dim=1)
        out = self.norm(out)
        self._cond_feature = self._film_gamma = self._film_beta = None
        return out


class StageBlock(nn.Module):
    def __init__(self, dim, num_groups=16, patch_size=8, transformer_heads=8, transformer_ffn_ratio=2.0,
                 transformer_dropout=0.0, proposal_scale_init=1.0):
        super().__init__()
        self.local_module1 = LocalModule(dim, num_groups=num_groups)
        self.mixed_module = MixedModule(
            dim,
            patch_size=patch_size,
            transformer_heads=transformer_heads,
            transformer_ffn_ratio=transformer_ffn_ratio,
            transformer_dropout=transformer_dropout,
            proposal_scale_init=proposal_scale_init,
        )
        self.local_module2 = LocalModule(dim, num_groups=num_groups)

    def forward(self, x, x_mask):
        x = self.local_module1(x, x_mask)
        x = self.mixed_module(x, x_mask)
        return self.local_module2(x, x_mask)
