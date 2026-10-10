import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from pyg_graph_transformer import GraphTransformerEncoder


class FeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingAdapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoleculePromptAdapter(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        out_dim,
        rank=32,
        prompt_enabled=True,
        dropout=0.1,
        input_dropout=0.0,
        scale_init=0.0,
        contrastive_enabled=False,
        contrastive_dropout=0.15,
        contrastive_temperature=0.2,
    ):
        super().__init__()
        self.prompt_enabled = bool(prompt_enabled)
        if self.prompt_enabled and rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = float(input_dropout)
        self.contrastive_enabled = bool(contrastive_enabled)
        self.contrastive_dropout = float(contrastive_dropout)
        self.contrastive_temperature = float(contrastive_temperature)

        self.base = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.prompt_enabled:
            self.prompt_down = nn.Linear(embedding_dim, rank, bias=False)
            self.prompt_up = nn.Linear(rank, out_dim, bias=False)
            self.prompt_dropout = nn.Dropout(dropout)
            self.prompt_scale = nn.Parameter(torch.tensor(float(scale_init)))
        else:
            self.prompt_down = None
            self.prompt_up = None
            self.prompt_dropout = None
            self.register_buffer('prompt_scale', torch.zeros(()))
        self.out_norm = nn.LayerNorm(out_dim)

    def _encode_normed(self, x):
        if self.training and self.input_dropout > 0:
            x = F.dropout(x, p=self.input_dropout, training=True)

        base_vec = self.base(x)
        if self.prompt_enabled:
            prompt_vec = self.prompt_up(self.prompt_dropout(F.gelu(self.prompt_down(x))))
            entity = self.out_norm(base_vec + self.prompt_scale * prompt_vec)
        else:
            prompt_vec = torch.zeros_like(base_vec)
            entity = self.out_norm(base_vec)
        return entity, base_vec, prompt_vec

    def _contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)
        if batch_size < 2:
            return z1.new_zeros(())

        temperature = max(self.contrastive_temperature, 1e-6)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.t()) / temperature
        targets = torch.arange(batch_size, device=z1.device)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.t(), targets)
        )

    def forward(self, x, return_aux=False):
        x = self.input_norm(x.float())
        entity, base_vec, prompt_vec = self._encode_normed(x)

        aux = None
        if return_aux:
            contrastive_loss = entity.new_zeros(())
            if self.training and self.contrastive_enabled:
                masked_x = F.dropout(
                    x,
                    p=self.contrastive_dropout,
                    training=True,
                )
                masked_entity, _, _ = self._encode_normed(masked_x)
                contrastive_loss = self._contrastive_loss(entity, masked_entity)

            aux = {
                'base_vec': base_vec,
                'prompt_vec': prompt_vec,
                'prompt_scale': self.prompt_scale,
                'prompt_enabled': self.prompt_enabled,
                'contrastive_loss': contrastive_loss,
            }

        return entity, aux


class PretrainedFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.float())


class DescriptorEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.feature_gate = nn.Parameter(torch.zeros(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        gate = torch.sigmoid(self.feature_gate)
        gated_x = x * gate
        return self.net(gated_x), gate


class GatedPairFusion(nn.Module):
    def __init__(self, left_dim, right_dim, out_dim, context_dim=0, dropout=0.1):
        super().__init__()
        gate_input_dim = left_dim + right_dim + context_dim
        self.left_proj = nn.Linear(left_dim, out_dim)
        self.right_proj = nn.Linear(right_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, left, right, context=None):
        left_h = self.left_proj(left)
        right_h = self.right_proj(right)

        gate_inputs = [left, right]
        if context is not None:
            gate_inputs.append(context)
        alpha = torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=1)))

        fused = alpha * left_h + (1.0 - alpha) * right_h
        return self.out(fused), alpha


class ResidualGatedFusion(nn.Module):
    """Add a new view without discarding an already useful base representation.

    The residual scale is initialized from the config (zero by default), so a
    newly enabled view starts with exactly zero contribution. This makes the
    pretrained-only model the initial state of the pretrained+graph variant.
    """

    def __init__(self, base_dim, residual_dim, out_dim, dropout=0.1, scale_init=0.0):
        super().__init__()
        self.base_proj = nn.Identity() if base_dim == out_dim else nn.Linear(base_dim, out_dim)
        self.residual_proj = nn.Sequential(
            nn.Linear(residual_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(base_dim + residual_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, base, residual):
        base_h = self.base_proj(base)
        residual_h = self.residual_proj(residual)
        sample_gate = torch.sigmoid(self.gate(torch.cat([base, residual], dim=1)))
        effective_gate = torch.tanh(self.residual_scale) * sample_gate
        return base_h + effective_gate * residual_h, effective_gate


class PairRelationBlock(nn.Module):
    def __init__(self, entity_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(entity_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, left, right):
        pair_feat = torch.cat(
            [left, right, left * right, torch.abs(left - right)],
            dim=1,
        )
        return self.net(pair_feat)


class TernaryRelationBlock(nn.Module):
    def __init__(self, entity_dim, out_dim, dropout=0.1):
        super().__init__()
        in_dim = entity_dim * 7
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_protac, h_poi, h_e3):
        tri_feat = torch.cat(
            [
                h_protac,
                h_poi,
                h_e3,
                h_protac * h_poi,
                h_protac * h_e3,
                h_poi * h_e3,
                h_protac * h_poi * h_e3,
            ],
            dim=1,
        )
        return self.net(tri_feat)


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RelationAttentionPooler(nn.Module):
    def __init__(self, token_dim, num_queries=1, heads=4, dropout=0.1):
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by heads={heads}")

        self.num_queries = num_queries
        self.query = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.normal_(self.query, mean=0.0, std=(2 * token_dim) ** -0.5)

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        query = self.query.unsqueeze(0).expand(tokens.size(0), -1, -1)
        pooled, attn_weights = self.attn(
            query,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        pooled = self.out(pooled)
        if self.num_queries == 1:
            pooled = pooled[:, 0]
        else:
            pooled = pooled.flatten(1)
        return pooled, attn_weights


class GraphAttentionPooler(nn.Module):
    def __init__(self, token_dim, heads=4, dropout=0.1):
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by heads={heads}")

        self.query = nn.Parameter(torch.empty(1, token_dim))
        nn.init.normal_(self.query, mean=0.0, std=(2 * token_dim) ** -0.5)

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, batch, fragment_id=None):
        dense_tokens, mask = to_dense_batch(tokens, batch)
        query = self.query.unsqueeze(0).expand(dense_tokens.size(0), -1, -1)
        pooled, attn_weights = self.attn(
            query,
            dense_tokens,
            dense_tokens,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=True,
        )
        pooled = self.out(pooled)
        return pooled[:, 0], attn_weights


class FragmentGraphPooler(nn.Module):
    def __init__(
        self,
        token_dim,
        heads=4,
        dropout=0.1,
        fragment_ids=(0, 1, 2),
        unknown_fragment_id=3,
        mode='residual',
        scale_init=0.0,
    ):
        super().__init__()
        valid_modes = {'concat', 'residual'}
        if mode not in valid_modes:
            raise ValueError(f"Unknown fragment pooling mode={mode}. Expected one of {sorted(valid_modes)}")

        self.mode = mode
        self.fragment_ids = tuple(fragment_ids)
        self.unknown_fragment_id = unknown_fragment_id
        self.whole_pooler = GraphAttentionPooler(
            token_dim=token_dim,
            heads=heads,
            dropout=dropout,
        )
        fragment_dim = token_dim * len(self.fragment_ids)
        if self.mode == 'concat':
            self.out = nn.Sequential(
                nn.Linear(token_dim + fragment_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fragment_delta = None
            self.fragment_scale = None
        else:
            self.out = None
            self.fragment_delta = nn.Sequential(
                nn.Linear(fragment_dim, token_dim, bias=False),
                nn.LayerNorm(token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(token_dim, token_dim, bias=False),
            )
            self.fragment_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _fragment_mean_pool(self, tokens, batch, fragment_id):
        dense_tokens, mask = to_dense_batch(tokens, batch)
        batch_size, _, token_dim = dense_tokens.shape

        if fragment_id is None:
            return tokens.new_zeros(batch_size, len(self.fragment_ids) * token_dim)

        dense_fragment, _ = to_dense_batch(
            fragment_id.view(-1, 1).long(),
            batch,
            fill_value=self.unknown_fragment_id,
        )
        dense_fragment = dense_fragment.squeeze(-1)

        pooled_fragments = []
        for current_fragment_id in self.fragment_ids:
            fragment_mask = mask & (dense_fragment == current_fragment_id)
            fragment_mask = fragment_mask.unsqueeze(-1)
            summed = (dense_tokens * fragment_mask).sum(dim=1)
            denom = fragment_mask.sum(dim=1).clamp_min(1)
            pooled_fragments.append(summed / denom)

        return torch.cat(pooled_fragments, dim=1)

    def forward(self, tokens, batch, fragment_id=None):
        whole_vec, attn_weights = self.whole_pooler(tokens, batch)
        fragment_vec = self._fragment_mean_pool(tokens, batch, fragment_id)
        if self.mode == 'concat':
            pooled = self.out(torch.cat([whole_vec, fragment_vec], dim=1))
        else:
            fragment_delta = self.fragment_delta(fragment_vec)
            pooled = whole_vec + self.fragment_scale * fragment_delta
        return pooled, attn_weights


class DeterministicSelfAttentionLayer(nn.Module):
    def __init__(self, dim, heads=4, ff_multiplier=2, dropout=0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_multiplier, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        batch_size, token_count, dim = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(batch_size, token_count, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, token_count, dim)

        x = x + self.out_dropout(self.out_proj(context))
        x = x + self.ffn(self.norm2(x))
        return x


class EntityInteractionBlock(nn.Module):
    def __init__(self, entity_dim, heads=4, layers=1, ff_multiplier=2, dropout=0.1):
        super().__init__()
        if entity_dim % heads != 0:
            raise ValueError(f"entity_dim={entity_dim} must be divisible by heads={heads}")

        self.role_tokens = nn.Parameter(torch.randn(3, entity_dim) * 0.02)
        self.layers = nn.ModuleList([
            DeterministicSelfAttentionLayer(
                dim=entity_dim,
                heads=heads,
                ff_multiplier=ff_multiplier,
                dropout=dropout,
            )
            for _ in range(layers)
        ])
        self.out_norm = nn.LayerNorm(entity_dim)

    def forward(self, h_protac, h_poi, h_e3):
        tokens = torch.stack([h_protac, h_poi, h_e3], dim=1)
        tokens = tokens + self.role_tokens.unsqueeze(0)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.out_norm(tokens)
        return tokens[:, 0], tokens[:, 1], tokens[:, 2]


class RelationInformationBottleneck(nn.Module):
    def __init__(
        self,
        token_dim,
        num_tokens=4,
        hidden_dim=None,
        dropout=0.1,
        temperature=1.0,
        use_noise=False,
        use_gumbel=False,
        eps=1e-6,
    ):
        super().__init__()
        hidden_dim = hidden_dim or token_dim

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.temperature = temperature
        self.use_noise = use_noise
        self.use_gumbel = use_gumbel
        self.eps = eps

        self.gate_net = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _sample_gate(self, logits):
        if self.training and self.use_gumbel:
            u = torch.rand_like(logits).clamp_(self.eps, 1.0 - self.eps)
            logistic_noise = torch.log(u) - torch.log(1.0 - u)
            return torch.sigmoid((logits + logistic_noise) / self.temperature)
        return torch.sigmoid(logits)

    def forward(self, relation_tokens):
        _, token_count, token_dim = relation_tokens.shape
        if token_count != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} relation tokens, but got {token_count}."
            )

        global_context = relation_tokens.mean(dim=1, keepdim=True).expand(
            -1,
            token_count,
            -1,
        )
        gate_input = torch.cat([relation_tokens, global_context], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate = self._sample_gate(gate_logits)

        if self.use_noise and self.training:
            batch_mean = relation_tokens.mean(dim=(0, 1), keepdim=True)
            batch_std = relation_tokens.std(
                dim=(0, 1),
                keepdim=True,
                unbiased=False,
            ).clamp_min(self.eps)
            noise = batch_mean + torch.randn_like(relation_tokens) * batch_std
            compact_tokens = gate * relation_tokens + (1.0 - gate) * noise
        else:
            compact_tokens = gate * relation_tokens

        sparsity_loss = gate.mean()

        a_term = ((1.0 - gate) ** 2).sum(dim=1).clamp_min(self.eps)
        token_mean = relation_tokens.mean(dim=1, keepdim=True)
        token_std = relation_tokens.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)
        normalized_tokens = (relation_tokens - token_mean) / token_std
        b_term = (gate * normalized_tokens).sum(dim=1)

        lmi_loss = (
            -0.5 * token_dim * torch.log(a_term).mean()
            + 0.5 * token_dim / token_count * a_term.mean()
            + 0.5 / token_count * b_term.pow(2).sum(dim=-1).mean()
        )

        aux = {
            'gate': gate,
            'sparsity_loss': sparsity_loss,
            'lmi_loss': lmi_loss,
        }
        return compact_tokens, aux


class PROTAC_STAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        protac_cfg = cfg['protac']
        protein_cfg = cfg['protein']
        clf_cfg = cfg['clf']
        descriptor_cfg = cfg.get('descriptor', {})
        fusion_cfg = cfg.get('fusion', {})
        views_cfg = cfg.get('views', {})
        interaction_cfg = cfg.get('interaction', {})
        relation_pool_cfg = cfg.get('relation_pooling', {})
        relation_ib_cfg = cfg.get('relation_ib', {})
        pretrained_mol_cfg = protac_cfg.get('pretrained_molecule', {})

        graph_dim = protac_cfg.get('hidden', 128)
        seq_dim = protein_cfg.get('out_dim', 64)
        saprot_dim = protein_cfg.get('saprot_embed', protein_cfg.get('embed', 1280))
        saprot_hidden = protein_cfg.get('saprot_hidden', protein_cfg.get('hidden', 128))
        protac_seq_hidden = protac_cfg.get('seq_hidden', protac_cfg.get('hidden', 128))
        entity_dim = fusion_cfg.get('entity_dim', graph_dim)
        relation_dim = fusion_cfg.get('relation_dim', entity_dim)
        dropout = fusion_cfg.get('dropout', 0.1)
        tri_proj_dim = cfg.get('tri_proj_dim', entity_dim)

        self.use_protac_graph = views_cfg.get('use_protac_graph', True)
        self.use_descriptors = views_cfg.get('use_descriptors', True)
        self.use_entity_attention = views_cfg.get('use_entity_attention', True)
        self.protac_feature_mode = protac_cfg.get('feature_mode', 'graph_seq')
        valid_protac_modes = {
            'graph_seq',
            'graph_only',
            'seq_only',
            'pretrained',
            'pretrained_graph',
        }
        if self.protac_feature_mode not in valid_protac_modes:
            raise ValueError(
                f"Unknown protac feature_mode={self.protac_feature_mode}. "
                f"Expected one of {sorted(valid_protac_modes)}"
            )
        if self.protac_feature_mode in {'graph_seq', 'graph_only', 'pretrained_graph'} and not self.use_protac_graph:
            raise ValueError(
                f"protac feature_mode={self.protac_feature_mode} requires use_protac_graph=true"
            )

        descriptor_in_dim = descriptor_cfg.get('in_dim', 8)
        descriptor_hidden = descriptor_cfg.get('hidden', 32)
        self.descriptor_out_dim = descriptor_cfg.get('out_dim', seq_dim)
        descriptor_dropout = descriptor_cfg.get('dropout', dropout)

        self.use_pretrained_molecule = pretrained_mol_cfg.get(
            'enabled',
            self.protac_feature_mode in {'pretrained', 'pretrained_graph'},
        )
        if (
            self.protac_feature_mode in {'pretrained', 'pretrained_graph'}
            and not self.use_pretrained_molecule
        ):
            raise ValueError(
                f"protac feature_mode={self.protac_feature_mode!r} "
                "requires pretrained_molecule.enabled=true"
            )

        if self.use_pretrained_molecule:
            self.protac_pretrained_encoder = MoleculePromptAdapter(
                embedding_dim=pretrained_mol_cfg.get('in_dim', 768),
                hidden_dim=pretrained_mol_cfg.get('hidden', max(entity_dim, 256)),
                out_dim=entity_dim,
                rank=pretrained_mol_cfg.get('rank', 32),
                prompt_enabled=pretrained_mol_cfg.get('prompt_enabled', True),
                dropout=pretrained_mol_cfg.get('dropout', dropout),
                input_dropout=pretrained_mol_cfg.get('input_dropout', 0.0),
                scale_init=pretrained_mol_cfg.get('scale_init', 0.0),
                contrastive_enabled=pretrained_mol_cfg.get('contrastive_enabled', False),
                contrastive_dropout=pretrained_mol_cfg.get('contrastive_dropout', 0.15),
                contrastive_temperature=pretrained_mol_cfg.get('contrastive_temperature', 0.2),
            )
        else:
            self.protac_pretrained_encoder = None

        if self.use_protac_graph:
            fragment_bias_cfg = protac_cfg.get('fragment_attention_bias', {})
            self.protac_encoder = GraphTransformerEncoder(
                node_dim=41,
                edge_dim=10,
                hidden_dim=graph_dim,
                out_dim=graph_dim,
                n_layers=protac_cfg.get('graph_layers', 1),
                n_heads=protac_cfg.get('graph_heads', 4),
                dropout=dropout,
                pos_enc_dim=8,
                fragment_bias_cfg=fragment_bias_cfg,
            )
            graph_pool_cfg = protac_cfg.get('graph_pooling', {})
            fragment_pool_cfg = protac_cfg.get('fragment_pooling', {})
            if fragment_pool_cfg.get('enabled', False):
                self.protac_graph_pooler = FragmentGraphPooler(
                    token_dim=graph_dim,
                    heads=graph_pool_cfg.get('heads', 4),
                    dropout=fragment_pool_cfg.get(
                        'dropout',
                        graph_pool_cfg.get('dropout', dropout),
                    ),
                    fragment_ids=fragment_pool_cfg.get('fragment_ids', [0, 1, 2]),
                    unknown_fragment_id=fragment_pool_cfg.get('unknown_fragment_id', 3),
                    mode=fragment_pool_cfg.get('mode', 'residual'),
                    scale_init=fragment_pool_cfg.get('scale_init', 0.0),
                )
            else:
                self.protac_graph_pooler = GraphAttentionPooler(
                    token_dim=graph_dim,
                    heads=graph_pool_cfg.get('heads', 4),
                    dropout=graph_pool_cfg.get('dropout', dropout),
                )
            self.protac_seq_graph_fuse = GatedPairFusion(
                left_dim=seq_dim,
                right_dim=graph_dim,
                out_dim=entity_dim,
                dropout=dropout,
            )
            if self.protac_feature_mode == 'pretrained_graph':
                pretrained_graph_fusion_cfg = protac_cfg.get('pretrained_graph_fusion', {})
                self.protac_pretrained_graph_fuse = ResidualGatedFusion(
                    base_dim=entity_dim,
                    residual_dim=graph_dim,
                    out_dim=entity_dim,
                    dropout=pretrained_graph_fusion_cfg.get('dropout', dropout),
                    scale_init=pretrained_graph_fusion_cfg.get('scale_init', 0.0),
                )
            else:
                self.protac_pretrained_graph_fuse = None
        else:
            self.protac_encoder = None
            self.protac_graph_pooler = None
            self.protac_seq_graph_fuse = None
            self.protac_pretrained_graph_fuse = None

        self.protein_saprot_encoder = PretrainedFeatureEncoder(
            embedding_dim=saprot_dim,
            hidden_dim=saprot_hidden,
            out_dim=entity_dim,
            dropout=dropout,
        )
        self.protac_seq_encoder = EmbeddingAdapter(
            embedding_dim=768,
            hidden_dim=protac_seq_hidden,
            out_dim=seq_dim,
            dropout=dropout,
        )

        self.protac_seq_only = FeatureProjector(seq_dim, entity_dim, dropout=dropout)
        self.protac_graph_only = FeatureProjector(graph_dim, entity_dim, dropout=dropout)

        if self.use_descriptors:
            self.descriptor_encoder = DescriptorEncoder(
                in_dim=descriptor_in_dim,
                hidden_dim=descriptor_hidden,
                out_dim=self.descriptor_out_dim,
                dropout=descriptor_dropout,
            )
            self.protac_descriptor_fuse = GatedPairFusion(
                left_dim=entity_dim,
                right_dim=self.descriptor_out_dim,
                out_dim=entity_dim,
                dropout=dropout,
            )
        else:
            self.descriptor_encoder = None
            self.protac_descriptor_fuse = None

        self.poi_role = nn.Parameter(torch.randn(1, entity_dim) * 0.02)
        self.e3_role = nn.Parameter(torch.randn(1, entity_dim) * 0.02)
        self.entity_norm = nn.LayerNorm(entity_dim)

        interaction_layers = interaction_cfg.get('layers', 1)
        if self.use_entity_attention and interaction_layers > 0:
            self.entity_interaction = EntityInteractionBlock(
                entity_dim=entity_dim,
                heads=interaction_cfg.get('heads', 4),
                layers=interaction_layers,
                ff_multiplier=interaction_cfg.get('ff_multiplier', 2),
                dropout=interaction_cfg.get('dropout', dropout),
            )
        else:
            self.entity_interaction = None

        self.protac_poi_relation = PairRelationBlock(entity_dim, relation_dim, dropout=dropout)
        self.protac_e3_relation = PairRelationBlock(entity_dim, relation_dim, dropout=dropout)
        self.poi_e3_relation = PairRelationBlock(entity_dim, relation_dim, dropout=dropout)
        self.ternary_relation = TernaryRelationBlock(
            entity_dim=entity_dim,
            out_dim=relation_dim,
            dropout=dropout,
        )

        self.poi_tri_proj = MLPProjector(entity_dim, tri_proj_dim, dropout=dropout)
        self.protac_tri_proj = MLPProjector(entity_dim, tri_proj_dim, dropout=dropout)
        self.e3_tri_proj = MLPProjector(entity_dim, tri_proj_dim, dropout=dropout)

        self.relation_token_names = ('protac_poi', 'protac_e3', 'poi_e3', 'ternary')

        self.use_relation_ib = relation_ib_cfg.get('enabled', False)
        if self.use_relation_ib:
            self.relation_ib = RelationInformationBottleneck(
                token_dim=relation_dim,
                num_tokens=len(self.relation_token_names),
                hidden_dim=relation_ib_cfg.get('hidden_dim', relation_dim),
                dropout=relation_ib_cfg.get('dropout', dropout),
                temperature=relation_ib_cfg.get('temperature', 1.0),
                use_noise=relation_ib_cfg.get('use_noise', False),
                use_gumbel=relation_ib_cfg.get('use_gumbel', False),
                eps=relation_ib_cfg.get('eps', 1e-6),
            )
        else:
            self.relation_ib = None

        self.use_relation_attention_pooling = relation_pool_cfg.get('enabled', True)
        relation_num_queries = relation_pool_cfg.get('num_queries', 1)
        if self.use_relation_attention_pooling:
            self.relation_pooler = RelationAttentionPooler(
                token_dim=relation_dim,
                num_queries=relation_num_queries,
                heads=relation_pool_cfg.get('heads', 4),
                dropout=relation_pool_cfg.get('dropout', dropout),
            )
            final_dim = relation_dim * relation_num_queries
        else:
            self.relation_pooler = None
            final_dim = relation_dim * len(self.relation_token_names)

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, clf_cfg['hidden']),
            nn.BatchNorm1d(clf_cfg['hidden']),
            nn.GELU(),
            nn.Dropout(clf_cfg.get('dropout', 0.3)),
            nn.Linear(clf_cfg['hidden'], clf_cfg['class']),
        )

    @staticmethod
    def masked_mean_pool(x, batch):
        dense_x, mask = to_dense_batch(x, batch)
        mask = mask.unsqueeze(-1)
        summed = (dense_x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return summed / denom

    @staticmethod
    def zero_feature(reference, out_dim):
        return reference.new_zeros(reference.size(0), out_dim)

    def encode_protac_entity(self, protac, protac_smile, descriptor_vec, return_aux=False):
        protac_pretrained_aux = None
        if self.protac_feature_mode in {'pretrained', 'pretrained_graph'}:
            if self.protac_pretrained_encoder is None:
                raise RuntimeError(
                    f"protac feature_mode={self.protac_feature_mode!r} "
                    "requires pretrained molecule encoder"
                )
            protac_entity, protac_pretrained_aux = self.protac_pretrained_encoder(
                protac_smile,
                return_aux=return_aux,
            )
            protac_graph_gate = None
            if self.protac_feature_mode == 'pretrained_graph':
                if self.protac_encoder is None or self.protac_pretrained_graph_fuse is None:
                    raise RuntimeError(
                        "protac feature_mode='pretrained_graph' requires use_protac_graph=true"
                    )
                protac_graph_tokens = self.protac_encoder(protac)
                protac_graph_vec, _ = self.protac_graph_pooler(
                    protac_graph_tokens,
                    protac.batch,
                    getattr(protac, 'fragment_id', None),
                )
                protac_entity, protac_graph_gate = self.protac_pretrained_graph_fuse(
                    protac_entity,
                    protac_graph_vec,
                )
        else:
            protac_seq_vec = self.protac_seq_encoder(protac_smile)
            protac_graph_vec = None

            protac_graph_gate = None
            if self.use_protac_graph or self.protac_feature_mode == 'graph_only':
                if self.protac_encoder is None:
                    raise RuntimeError(
                        f"protac feature_mode={self.protac_feature_mode} requires use_protac_graph=true"
                    )
                protac_graph_tokens = self.protac_encoder(protac)
                protac_graph_vec, _ = self.protac_graph_pooler(
                    protac_graph_tokens,
                    protac.batch,
                    getattr(protac, 'fragment_id', None),
                )

            if self.protac_feature_mode == 'graph_seq':
                protac_entity, protac_graph_gate = self.protac_seq_graph_fuse(
                    protac_seq_vec,
                    protac_graph_vec,
                )
            elif self.protac_feature_mode == 'graph_only':
                protac_entity = self.protac_graph_only(protac_graph_vec)
            elif self.protac_feature_mode == 'seq_only':
                protac_entity = self.protac_seq_only(protac_seq_vec)
            else:
                raise RuntimeError(f"Unhandled protac feature_mode={self.protac_feature_mode}")

        protac_desc_gate = None
        if self.use_descriptors:
            protac_entity, protac_desc_gate = self.protac_descriptor_fuse(protac_entity, descriptor_vec)

        return protac_entity, protac_graph_gate, protac_desc_gate, protac_pretrained_aux

    def encode_protein_entity(self, saprot_tensor, role_embedding):
        entity = self.protein_saprot_encoder(saprot_tensor)
        entity = self.entity_norm(entity + role_embedding.expand(entity.size(0), -1))
        return entity

    def forward(
        self,
        protac,
        poi_saprot,
        e3_saprot,
        protac_smile,
        descriptor,
        return_aux=False,
    ):
        if self.use_descriptors:
            descriptor_vec, descriptor_gate = self.descriptor_encoder(descriptor)
        else:
            descriptor_vec = self.zero_feature(protac_smile, self.descriptor_out_dim)
            descriptor_gate = None

        h_protac, protac_graph_gate, protac_desc_gate, protac_pretrained_aux = self.encode_protac_entity(
            protac,
            protac_smile,
            descriptor_vec,
            return_aux=return_aux,
        )
        h_poi = self.encode_protein_entity(poi_saprot, self.poi_role)
        h_e3 = self.encode_protein_entity(e3_saprot, self.e3_role)

        if self.entity_interaction is not None:
            h_protac, h_poi, h_e3 = self.entity_interaction(
                h_protac,
                h_poi,
                h_e3,
            )

        r_protac_poi = self.protac_poi_relation(h_protac, h_poi)
        r_protac_e3 = self.protac_e3_relation(h_protac, h_e3)
        r_poi_e3 = self.poi_e3_relation(h_poi, h_e3)
        r_tri = self.ternary_relation(h_protac, h_poi, h_e3)

        relation_tokens = torch.stack([r_protac_poi, r_protac_e3, r_poi_e3, r_tri], dim=1)

        relation_ib_aux = None
        if self.relation_ib is not None:
            relation_tokens, relation_ib_aux = self.relation_ib(relation_tokens)

        relation_attention = None
        if self.relation_pooler is not None:
            classifier_input, relation_attention = self.relation_pooler(relation_tokens)
        else:
            classifier_input = relation_tokens.flatten(1)

        logits = self.classifier(classifier_input)

        if return_aux:
            aux = {
                'entity_feats': {
                    'poi': h_poi,
                    'protac': h_protac,
                    'e3': h_e3,
                },
                'tri_feats': {
                    'poi': self.poi_tri_proj(h_poi),
                    'protac': self.protac_tri_proj(h_protac),
                    'e3': self.e3_tri_proj(h_e3),
                },
                'relation_feats': {
                    'protac_poi': r_protac_poi,
                    'protac_e3': r_protac_e3,
                    'poi_e3': r_poi_e3,
                    'ternary': r_tri,
                },
                'gates': {
                    'descriptor': descriptor_gate,
                    'protac_graph': protac_graph_gate,
                    'protac_descriptor': protac_desc_gate,
                },
                'protac_pretrained': protac_pretrained_aux,
                'relation_attention': relation_attention,
                'relation_ib': relation_ib_aux,
                'relation_token_names': self.relation_token_names,
            }
            return logits, aux

        return logits
