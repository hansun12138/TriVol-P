# pyg_graph_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=True,
        batch_norm=False,
        residual=True,
        use_bias=False,
        use_fragment_bias=False,
        num_fragment_types=4,
        unknown_fragment_id=3,
        fragment_bias_init=0.0,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.use_fragment_bias = use_fragment_bias
        self.num_fragment_types = num_fragment_types
        self.unknown_fragment_id = unknown_fragment_id

        # 确保维度对齐
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads

        # 1. Attention Layers
        self.Q = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim, bias=use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.use_fragment_bias:
            self.fragment_pair_bias = nn.Parameter(
                torch.full(
                    (num_fragment_types, num_fragment_types, num_heads),
                    float(fragment_bias_init),
                )
            )
        else:
            self.register_parameter('fragment_pair_bias', None)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        # 2. FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # 3. FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q.weight)
        glorot(self.K.weight)
        glorot(self.V.weight)
        glorot(self.proj_e.weight)
        glorot(self.O_h.weight)
        glorot(self.O_e.weight)

    def _fragment_bias(self, fragment_id, src, dst):
        if (
            not self.use_fragment_bias
            or self.fragment_pair_bias is None
            or fragment_id is None
            or src.numel() == 0
        ):
            return None

        fragment_id = fragment_id.to(device=src.device, dtype=torch.long)
        src_fragment = fragment_id[src]
        dst_fragment = fragment_id[dst]

        valid_pair = (
            (src_fragment >= 0)
            & (src_fragment < self.num_fragment_types)
            & (dst_fragment >= 0)
            & (dst_fragment < self.num_fragment_types)
            & (src_fragment != self.unknown_fragment_id)
            & (dst_fragment != self.unknown_fragment_id)
        )

        safe_src_fragment = src_fragment.clamp(0, self.num_fragment_types - 1)
        safe_dst_fragment = dst_fragment.clamp(0, self.num_fragment_types - 1)
        bias = self.fragment_pair_bias[safe_src_fragment, safe_dst_fragment]
        return bias * valid_pair.unsqueeze(-1).to(bias.dtype)

    def forward(self, x, edge_index, edge_attr, fragment_id=None):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E, in_dim]
        fragment_id: [N], optional atom-level fragment ids
        """
        h_in1 = x
        e_in1 = edge_attr

        # --- 1. 计算 Q, K, V, E ---
        # [N, Heads, Head_Dim]
        Q_h = self.Q(x).view(-1, self.num_heads, self.head_dim)
        K_h = self.K(x).view(-1, self.num_heads, self.head_dim)
        V_h = self.V(x).view(-1, self.num_heads, self.head_dim)
        # [E, Heads, Head_Dim]
        proj_e = self.proj_e(edge_attr).view(-1, self.num_heads, self.head_dim)

        # --- 2. 手动实现 Attention 逻辑 (复刻 DGL src_dot_dst & imp_exp_attn) ---
        src, dst = edge_index

        # 获取源节点和目标节点的 Q, K
        # 注意: DGL代码中 src_dot_dst 是 src * dst
        # PyG 的 message passing 默认是从 src -> dst (j -> i)
        q_dst = Q_h[dst]  # [E, Heads, Dim]
        k_src = K_h[src]  # [E, Heads, Dim]
        v_src = V_h[src]  # [E, Heads, Dim]

        # Score 计算: (Q * K) / sqrt(d)
        # 这里使用逐元素相乘 (Multiplicative Attention)，不是点积
        scores = (q_dst * k_src) / (self.head_dim ** 0.5)

        # 融合边特征: Score * Proj_e
        # 这就是新的边特征 (e_out) 的前身
        e_out_raw = scores * proj_e  # [E, Heads, Dim]

        # 计算注意力权重: Sum over dim -> Softmax
        # [E, Heads]
        attn_logits = e_out_raw.sum(dim=-1)
        fragment_bias = self._fragment_bias(fragment_id, src, dst)
        if fragment_bias is not None:
            attn_logits = attn_logits + fragment_bias
        attn_logits = torch.clamp(attn_logits, min=-5, max=5)
        attn_weights = softmax(attn_logits, index=dst, dim=0)  # [E, Heads]

        # 聚合节点特征: Sum(V * attn)
        msg = v_src * attn_weights.unsqueeze(-1)  # [E, Heads, Dim]
        h_attn_out = torch.zeros_like(Q_h)  # [N, Heads, Dim]
        # 将消息聚合到目标节点 dst
        h_attn_out.index_add_(0, dst, msg)

        # --- 3. 整理输出 ---
        h = h_attn_out.view(-1, self.out_channels)
        e = e_out_raw.view(-1, self.out_channels)  # 边特征变回 [E, Out_Dim]

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        # Residual & Norm 1
        if self.residual:
            h = h_in1 + h
            e = e_in1 + e
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        h_in2 = h
        e_in2 = e

        # --- 4. FFN ---
        # h FFN
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # e FFN
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        # Residual & Norm 2
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e
        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        return h, e


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        out_dim,
        n_layers,
        n_heads,
        dropout=0.2,
        pos_enc_dim=8,
        fragment_bias_cfg=None,
    ):
        super().__init__()
        fragment_bias_cfg = fragment_bias_cfg or {}
        use_fragment_bias = fragment_bias_cfg.get('enabled', False)
        num_fragment_types = fragment_bias_cfg.get('num_fragment_types', 4)
        unknown_fragment_id = fragment_bias_cfg.get('unknown_fragment_id', 3)
        fragment_bias_init = fragment_bias_cfg.get('init', 0.0)

        # Embedding Layers
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(0.0)

        # Stack GT Layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                use_fragment_bias=use_fragment_bias,
                num_fragment_types=num_fragment_types,
                unknown_fragment_id=unknown_fragment_id,
                fragment_bias_init=fragment_bias_init,
            ) for _ in range(n_layers - 1)
        ])

        # Last Layer
        self.layers.append(
            GraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_heads=n_heads,
                dropout=dropout,
                use_fragment_bias=use_fragment_bias,
                num_fragment_types=num_fragment_types,
                unknown_fragment_id=unknown_fragment_id,
                fragment_bias_init=fragment_bias_init,
            )
        )

    def forward(self, data):
        # 解包 PyG Data 对象
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        fragment_id = getattr(data, 'fragment_id', None)

        # 处理位置编码 (兼容可能没有 lap_pos_enc 的情况)
        if hasattr(data, 'lap_pos_enc') and data.lap_pos_enc is not None:
            lap_pos_enc = data.lap_pos_enc.float()
        else:
            # 如果没有位置编码，用全0替代，防止报错
            lap_pos_enc = torch.zeros(x.size(0), self.embedding_lap_pos_enc.in_features, device=x.device)

        # Input Embedding
        h = self.linear_h(x)
        pe = self.embedding_lap_pos_enc(lap_pos_enc)
        h = h + pe
        h = self.in_feat_dropout(h)

        e = self.linear_e(edge_attr)

        # Pass through layers
        for conv in self.layers:
            h, e = conv(h, edge_index, e, fragment_id=fragment_id)

        return h
