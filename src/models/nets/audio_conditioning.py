"""Speech conditioning network (DGN).

Encodes the speech features (HuBERT semantics, rhythm, mel), the CLIP transcript embedding,
the speaker identity (speaker id + SMPL-X body shape) and, for the editable model, the root
trajectory and keyposes, and turns them into per-stage features and FiLM parameters for the
denoiser trunk.

Two fusion modes (configs/dgn/):
  staged  (DGN v1)  staged dynamic gating: acoustic gate, semantic gates,
                    semantic bridge and a final gate.
  dgn_v2  (DGN v2)  semantic base + sparse residual gates: the HuBERT embedding plus the
                    identity forms the base, rhythm / mel / CLIP text are added through
                    per-frame sigmoid gates that start closed.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioFeatureEncoder(nn.Module):
    """Per-frame encoders for rhythm (3), HuBERT semantics (1024) and mel (128)."""

    def __init__(self, embed_dim: int, rhythm_in_dim: int = 3, semantic_in_dim: int = 1024, mel_in_dim: int = 128):
        super().__init__()
        self.rhythm_encoder = nn.Sequential(
            nn.Linear(rhythm_in_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_in_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.mel_encoder = nn.Sequential(
            nn.Linear(mel_in_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, rhythm, semantic, mel):
        return self.rhythm_encoder(rhythm), self.semantic_encoder(semantic), self.mel_encoder(mel)


class FusionGate(nn.Module):
    """Softmax gate over a group of modalities, fused into one embedding (DGN v1)."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modalities),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def weights(self, concat: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(concat), dim=-1)

    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(modality_embeddings, dim=-1)
        weights = self.weights(concat)
        weighted = torch.cat(
            [emb * weights[..., i : i + 1] for i, emb in enumerate(modality_embeddings)],
            dim=-1,
        )
        return self.fusion_proj(weighted)


class SigmoidFusionGate(FusionGate):
    """Independent sigmoid weight per modality (no competition); the final gate of DGN v1."""

    def weights(self, concat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(concat))


class PatchAttentionPool(nn.Module):
    """Learned frame-to-patch pooling, initialized to exact uniform averaging."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        hidden_dim = max(16, embed_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, e = x.shape
        patches = x.view(b, t // self.patch_size, self.patch_size, e)
        weights = torch.softmax(self.score(patches), dim=2)
        return (patches * weights).sum(dim=2)


class SemanticBridge(nn.Module):
    """Cross-attention from the HuBERT stream to the transcript semantics, with residuals (DGN v1)."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hubert_core: torch.Tensor, semantic_core: torch.Tensor) -> torch.Tensor:
        bridged, _ = self.cross_attn(query=hubert_core, key=semantic_core, value=semantic_core, need_weights=False)
        x = self.norm1(hubert_core + bridged)
        return self.norm2(x + self.ffn(x))


class AudioConditioning(nn.Module):
    def __init__(
        self,
        stage_dims: List[int],
        patch_size: int,
        base_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        context_dropout: float = 0.0,
        film_dropout: float = 0.0,
        rhythm_in_dim: int = 3,
        mel_in_dim: int = 128,
        semantic_in_dim: int = 1024,
        clip_text_dim: int = 512,
        activity_dim: int = 1,
        shape_dim: int = 10,
        num_speakers: int = 30,
        use_trajectory_condition: bool = False,
        trajectory_cond_dim: int = 4,
        use_keypose_condition: bool = False,
        keypose_cond_dim: int = 1320,
        fusion_mode: str = "dgn_v2",
        condition_gate_use_state: bool = False,
        condition_attention_pooling: bool = False,
        gate_bias_init: float = -3.0,
        frame_rhythm_bypass: bool = False,
        timestep_gate: bool = False,
        use_depth_routing: bool = False,
        num_train_timesteps: int = 1000,
    ):
        """
        stage_dims / patch_size / base_dim  trunk layout of the denoiser
        embed_dim / hidden_dim              width of the modality embeddings / gate MLPs
        activity_dim                        leading entries of `activity` used (activity[0] = speaker id / 30)
        use_trajectory_condition            root trajectory plan [B, T, 4] (editable model)
        use_keypose_condition               sparse keypose targets in the wavelet domain (editable model)
        fusion_mode                         "staged" (DGN v1) or "dgn_v2"
        DGN v2 components:
          condition_gate_use_state          gates also see the noisy motion state and the timestep
          condition_attention_pooling       learned frame-to-patch pooling instead of averaging
          gate_bias_init                    initial gate logit (sigmoid(-3) ~ 0.05: gates start closed)
          frame_rhythm_bypass               rhythm injected at frame resolution, bypassing patch pooling
          timestep_gate                     gate logits shifted by a learned per-modality noise-level term
          use_depth_routing                 each trunk stage mixes the base and gated streams with its own weights
        """
        super().__init__()
        if fusion_mode not in {"staged", "dgn_v2"}:
            raise ValueError(f"fusion_mode must be 'staged' or 'dgn_v2', got {fusion_mode!r}")
        self.stage_dims = stage_dims
        self.patch_size = patch_size
        self.base_dim = base_dim
        self.activity_dim = int(activity_dim)
        self.shape_dim = int(shape_dim)
        self.num_speakers = int(num_speakers)
        self.use_trajectory_condition = bool(use_trajectory_condition)
        self.use_keypose_condition = bool(use_keypose_condition)
        self.fusion_mode = fusion_mode
        self.condition_gate_use_state = bool(condition_gate_use_state)
        self.num_train_timesteps = int(num_train_timesteps)

        self.audio_encoder = AudioFeatureEncoder(embed_dim, rhythm_in_dim, semantic_in_dim, mel_in_dim)
        self.clip_text_encoder = nn.Sequential(
            nn.Linear(clip_text_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.trajectory_cond_encoder = None
        if self.use_trajectory_condition:
            self.trajectory_cond_encoder = nn.Sequential(
                nn.Linear(trajectory_cond_dim, embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        self.keypose_cond_encoder = None
        if self.use_keypose_condition:
            self.keypose_cond_encoder = nn.Sequential(
                nn.Linear(keypose_cond_dim * 2, embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

        hidden_dim = hidden_dim or base_dim
        if self.condition_gate_use_state:
            self.motion_gate_proj = nn.Linear(base_dim, embed_dim)
            self.timestep_gate_proj = nn.Linear(base_dim, embed_dim)
            self.gate_context_norm = nn.LayerNorm(embed_dim)

        # Identity: speaker id and body shape, plus a learned embedding per speaker
        # (row 0 is reserved for an unknown speaker).
        self.identity_proj = nn.Sequential(
            nn.Linear(self.activity_dim + self.shape_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.speaker_embedding = nn.Embedding(self.num_speakers + 1, embed_dim)
        nn.init.normal_(self.speaker_embedding.weight, std=0.02)

        if fusion_mode == "staged":
            # The semantic gates keep their original four slots (transcript, emotion, word
            # labels, word ids); only the transcript is used, the other slots are zero.
            self.acoustic_gate = FusionGate(embed_dim, hidden_dim, 2)
            self.global_sem_gate = FusionGate(embed_dim, hidden_dim, 2)
            self.local_sem_gate = FusionGate(embed_dim, hidden_dim, 2)
            self.semantic_merge_gate = FusionGate(embed_dim, hidden_dim, 2)
            self.semantic_bridge = SemanticBridge(embed_dim, hidden_dim, num_heads=8 if embed_dim % 8 == 0 else 4)
            self.final_gate = SigmoidFusionGate(embed_dim, hidden_dim, 4)
        else:
            self.residual_modality_names = ["rhythm", "mel", "clip_text"]
            num_residual = len(self.residual_modality_names)
            # LayerNorm only on the gate input, so that gates do not select by magnitude.
            self.gate_input_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in self.residual_modality_names])
            gate_in_dim = embed_dim * num_residual + (embed_dim if self.condition_gate_use_state else 0)
            gate_hidden = max(32, hidden_dim // 4)
            self.modality_gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, num_residual),
            )
            nn.init.zeros_(self.modality_gate_net[-1].weight)
            nn.init.constant_(self.modality_gate_net[-1].bias, float(gate_bias_init))
            self.gate_timestep_coeff = nn.Parameter(torch.zeros(num_residual)) if timestep_gate else None
            self.frame_rhythm_proj = None
            if frame_rhythm_bypass:
                self.frame_rhythm_proj = nn.Linear(embed_dim, base_dim)
                nn.init.zeros_(self.frame_rhythm_proj.weight)
                nn.init.zeros_(self.frame_rhythm_proj.bias)
            self.stage_routing_logits = (
                nn.Parameter(torch.zeros(len(stage_dims), num_residual + 1)) if use_depth_routing else None
            )

        self.feature_norm = nn.LayerNorm(embed_dim)
        self.patch_attention_pool = PatchAttentionPool(embed_dim, patch_size) if condition_attention_pooling else None
        # learned "null" condition for classifier-free guidance
        self.null_fused_feature = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.context_in = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.context_proj = nn.Conv1d(hidden_dim, base_dim, kernel_size=1)
        self.context_norm = nn.LayerNorm(base_dim)
        self.context_dropout = nn.Dropout(context_dropout)
        self.stage_feature_projs = nn.ModuleList(
            [nn.Sequential(nn.Linear(base_dim, base_dim), nn.SiLU(), nn.Linear(base_dim, dim)) for dim in stage_dims]
        )
        self.stage_film_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(base_dim, base_dim), nn.SiLU(), nn.Linear(base_dim, dim * 2)) for dim in stage_dims]
        )
        self.film_dropout = nn.Dropout(film_dropout)

    def forward(
        self,
        speech: Dict[str, torch.Tensor],
        full_mask: torch.Tensor,
        activity: torch.Tensor,
        cond_drop_mask: Optional[torch.Tensor] = None,
        clip_text: Optional[torch.Tensor] = None,
        shape_betas: Optional[torch.Tensor] = None,
        trajectory_cond: Optional[torch.Tensor] = None,
        keypose_hint: Optional[torch.Tensor] = None,
        keypose_mask: Optional[torch.Tensor] = None,
        motion_context: Optional[torch.Tensor] = None,
        timestep_context: Optional[torch.Tensor] = None,
        timestep_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        speech           {"rhythm": [B, T, 3], "semantic": [B, T, 1024], "mel": [B, T, 128]}
        full_mask        [B, 1 + T] (time token + motion frames)
        activity         [B, activity_dim]
        cond_drop_mask   [B] bool, samples whose condition is replaced by the null condition (CFG)
        clip_text        [B, 512]
        shape_betas      [B, shape_dim] normalized SMPL-X shape
        trajectory_cond  [B, T, 4] root trajectory plan (editable model)
        keypose_hint / keypose_mask  [B, T, D] keypose targets and soft mask (editable model)
        motion_context   [B, T, base_dim] encoded noisy motion; timestep_context [B, base_dim];
        timestep_values  [B] diffusion timesteps (DGN v2 gate inputs)

        Returns the per-stage conditions, plus the DGN v2 gates, the frame-level rhythm term
        and the global identity embedding when present.
        """
        expected_len = full_mask.shape[1] - 1
        rhythm = self._align_length(speech["rhythm"], expected_len)
        semantic = self._align_length(speech["semantic"], expected_len)
        mel = self._align_length(speech["mel"], expected_len)
        B, T = rhythm.shape[0], rhythm.shape[1]
        patch_pad = (self.patch_size - (T % self.patch_size)) % self.patch_size

        rhythm_emb, semantic_emb, mel_emb = self.audio_encoder(rhythm, semantic, mel)
        E = rhythm_emb.shape[-1]
        device, dtype = rhythm_emb.device, rhythm_emb.dtype

        gate_context = None
        if self.condition_gate_use_state:
            motion_context = self._align_length(motion_context.to(device=device, dtype=dtype), T)
            timestep_context = timestep_context.to(device=device, dtype=dtype).unsqueeze(1).expand(B, T, -1)
            gate_context = self.gate_context_norm(
                self.motion_gate_proj(motion_context) + self.timestep_gate_proj(timestep_context)
            )

        def zero_emb() -> torch.Tensor:
            return torch.zeros(B, T, E, device=device, dtype=dtype)

        # identity: [speaker id, body shape] -> embedding, plus the per-speaker embedding
        activity_vec = activity.to(device=device, dtype=dtype)
        activity_expanded = activity_vec.unsqueeze(1).expand(B, T, self.activity_dim)
        if shape_betas is None:
            shape_expanded = torch.zeros(B, T, self.shape_dim, device=device, dtype=dtype)
        else:
            shape_expanded = shape_betas.to(device=device, dtype=dtype).unsqueeze(1).expand(B, T, self.shape_dim)
        identity_emb = self.identity_proj(torch.cat([activity_expanded, shape_expanded], dim=-1))
        sid = torch.round(activity_vec[:, 0] * self.num_speakers).long().clamp(0, self.num_speakers)
        identity_emb = identity_emb + self.speaker_embedding(sid).to(dtype=identity_emb.dtype).unsqueeze(1)

        clip_text_emb = zero_emb()
        if clip_text is not None:
            clip_text_emb = self.clip_text_encoder(clip_text.to(dtype=dtype)).unsqueeze(1).expand(B, T, E)

        trajectory_cond_emb = None
        if self.trajectory_cond_encoder is not None:
            if trajectory_cond is None:
                trajectory_cond_emb = zero_emb()
            else:
                trajectory_cond_emb = self.trajectory_cond_encoder(
                    self._align_length(trajectory_cond, T).to(device=device, dtype=dtype)
                )
        keypose_emb = None
        if self.keypose_cond_encoder is not None:
            if keypose_hint is None or keypose_mask is None:
                keypose_emb = zero_emb()
            else:
                hint = self._align_length(keypose_hint, T).to(device=device, dtype=dtype)
                mask = self._align_length(keypose_mask, T).to(device=device, dtype=dtype).clamp(0.0, 1.0)
                keypose_emb = self.keypose_cond_encoder(torch.cat([hint * mask, mask], dim=-1))
                keypose_emb = keypose_emb * mask.abs().amax(dim=-1, keepdim=True).clamp(0.0, 1.0)

        modality_gates = None
        frame_conditioning = None
        source_streams = None
        if self.fusion_mode == "dgn_v2":
            # fused = base(semantic + identity) + sum_m g_m(t) * e_m(t)
            base_emb = semantic_emb + identity_emb
            res_list = [rhythm_emb, mel_emb, clip_text_emb]
            gate_in = torch.cat([norm(e) for norm, e in zip(self.gate_input_norms, res_list)], dim=-1)
            if self.condition_gate_use_state:
                gate_in = torch.cat([gate_in, gate_context], dim=-1)
            gate_logits = self.modality_gate_net(gate_in)  # [B, T, M]
            if self.gate_timestep_coeff is not None and timestep_values is not None:
                t_norm = timestep_values.to(device=device, dtype=gate_logits.dtype)
                t_norm = (t_norm / float(self.num_train_timesteps)).view(-1, 1, 1)
                gate_logits = gate_logits + self.gate_timestep_coeff.view(1, 1, -1) * (t_norm - 0.5)
            modality_gates = torch.sigmoid(gate_logits)
            # CFG unconditional branch: null base, all residual gates closed
            if cond_drop_mask is not None:
                drop = cond_drop_mask.view(-1, 1, 1).to(base_emb.device)
                null_feat = self.null_fused_feature.expand(base_emb.size(0), base_emb.size(1), -1)
                base_emb = torch.where(drop, null_feat.to(dtype=base_emb.dtype), base_emb)
                modality_gates = modality_gates * (~drop).to(modality_gates.dtype)
            gated_residuals = [modality_gates[..., i : i + 1] * e for i, e in enumerate(res_list)]
            fused_feature = base_emb
            for gated in gated_residuals:
                fused_feature = fused_feature + gated
            if trajectory_cond_emb is not None:
                fused_feature = fused_feature + trajectory_cond_emb
            if keypose_emb is not None:
                fused_feature = fused_feature + keypose_emb
            fused_feature = self.feature_norm(fused_feature)
            if self.frame_rhythm_proj is not None:
                frame_conditioning = modality_gates[..., 0:1] * self.frame_rhythm_proj(rhythm_emb)
            if self.stage_routing_logits is not None:
                source_streams = [base_emb] + gated_residuals
        else:
            acoustic_core = self.acoustic_gate([rhythm_emb, mel_emb])
            global_sem = self.global_sem_gate([clip_text_emb, zero_emb()])
            local_sem = self.local_sem_gate([zero_emb(), zero_emb()])
            semantic_core = self.semantic_merge_gate([global_sem, local_sem])
            hubert_sem = self.semantic_bridge(semantic_emb, semantic_core)
            fused_feature = self.final_gate([acoustic_core, hubert_sem, semantic_core, identity_emb])
            if trajectory_cond_emb is not None:
                fused_feature = fused_feature + trajectory_cond_emb
            if keypose_emb is not None:
                fused_feature = fused_feature + keypose_emb
            fused_feature = self.feature_norm(fused_feature)
            if cond_drop_mask is not None:
                drop = cond_drop_mask.view(-1, 1, 1).to(fused_feature.device)
                null_feat = self.null_fused_feature.expand(fused_feature.size(0), fused_feature.size(1), -1)
                fused_feature = torch.where(drop, null_feat, fused_feature)

        def pool_frames(feat: torch.Tensor) -> torch.Tensor:
            if patch_pad > 0:
                feat = torch.cat([feat, feat[:, -1:, :].repeat(1, patch_pad, 1)], dim=1)
            if self.patch_attention_pool is not None:
                return self.patch_attention_pool(feat)
            return F.avg_pool1d(feat.transpose(1, 2), kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        def context_from_patches(patches: torch.Tensor) -> torch.Tensor:
            ctx = self.context_proj(self.context_in(patches.transpose(1, 2))).transpose(1, 2)
            return self.context_dropout(self.context_norm(ctx))

        stage_conditions: List[Dict[str, torch.Tensor]] = []
        if source_streams is not None:
            # depth routing: each stage mixes the pooled base and gated streams with its own weights
            pooled_sources = torch.stack([pool_frames(stream) for stream in source_streams], dim=2)  # [B, P, S, E]
            route_weights = torch.softmax(self.stage_routing_logits, dim=-1)
            for idx in range(len(self.stage_dims)):
                ctx = context_from_patches((pooled_sources * route_weights[idx].view(1, 1, -1, 1)).sum(dim=2))
                gamma, beta = self.film_dropout(self.stage_film_generators[idx](ctx)).chunk(2, dim=-1)
                stage_conditions.append(
                    {"feature": self.stage_feature_projs[idx](ctx), "film_gamma": gamma, "film_beta": beta}
                )
        else:
            ctx = context_from_patches(pool_frames(fused_feature))
            for idx in range(len(self.stage_dims)):
                feature = self.stage_feature_projs[idx](ctx)
                gamma, beta = self.film_dropout(self.stage_film_generators[idx](ctx)).chunk(2, dim=-1)
                stage_conditions.append({"feature": feature, "film_gamma": gamma, "film_beta": beta})

        result = {"stage_conditions": stage_conditions}
        if modality_gates is not None:
            result["modality_gates"] = modality_gates
            result["modality_gate_names"] = list(self.residual_modality_names)
        if frame_conditioning is not None:
            result["frame_conditioning"] = frame_conditioning
        # Constant over time; taken before the CFG null replacement, so the unconditional
        # branch keeps the identity and guidance acts on the speech content only.
        result["identity_global"] = identity_emb[:, 0]
        return result

    @staticmethod
    def _align_length(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len, :]
        return torch.cat([tensor, tensor[:, -1:, :].repeat(1, target_len - tensor.shape[1], 1)], dim=1)
