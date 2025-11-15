from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax


class HANConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        concat: bool = True,
        negative_slope=0.2,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.concat = concat
        self.heads = heads
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.metadata = metadata
        self.use_layernorm = use_layernorm
        self.final_out_dim = out_channels if concat else out_channels // heads

        if not isinstance(in_channels, dict):
            in_channels = {ntype: in_channels for ntype in metadata[0]}
        self.in_channels = in_channels

        self.proj = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for ntype, in_ch in self.in_channels.items():
            self.proj[ntype] = Linear(in_ch, out_channels)
            if use_layernorm:
                self.norm[ntype] = nn.LayerNorm([heads, out_channels // heads])

        dim = out_channels // heads
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        for edge_type in metadata[1]:
            etype_str = '__'.join(edge_type)
            self.lin_src[etype_str] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[etype_str] = nn.Parameter(torch.empty(1, heads, dim))

        self.edge_encoder = nn.ModuleDict()

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        for param in self.lin_src.values():
            glorot(param)
        for param in self.lin_dst.values():
            glorot(param)
        for layer in self.edge_encoder.values():
            reset(layer)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        for ntype, x in x_dict.items():
            x_proj = self.proj[ntype](x).view(-1, H, D)
            if self.use_layernorm:
                x_proj = self.norm[ntype](x_proj)
            x_node_dict[ntype] = x_proj
            out_dict[ntype] = []

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            etype_str = '__'.join(edge_type)

            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            lin_src = self.lin_src[etype_str]
            lin_dst = self.lin_dst[etype_str]

            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)

            edge_attr = None
            if edge_attr_dict and edge_type in edge_attr_dict:
                edge_attr = edge_attr_dict[edge_type]
                if edge_attr is not None:
                    if etype_str not in self.edge_encoder:
                        edge_dim = edge_attr.size(-1)
                        self.edge_encoder[etype_str] = nn.Linear(edge_dim, self.out_channels)

            self.current_edge_type_str = etype_str

            out = self.propagate(
                edge_index=edge_index,
                x=(x_src, x_dst),
                alpha=(alpha_src, alpha_dst),
                edge_attr=edge_attr,
                size=(x_src.size(0), x_dst.size(0))
            )
            out = F.relu(out)
            out_dict[dst_type].append(out)

        for ntype, outs in out_dict.items():
            if outs:
                out = torch.stack(outs, dim=-1).mean(dim=-1)
                out_dict[ntype] = (
                    out.view(-1, H * D) if self.concat else out.mean(dim=1)
                )
            else:
                out_dict[ntype] = None

        return out_dict

    def message(self, x_j, alpha_i, alpha_j, edge_attr, index, ptr, size_i):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha - alpha.mean(dim=0, keepdim=True)
        alpha = torch.clamp(alpha, min=-10, max=10)
        alpha = torch.nan_to_num(alpha)

        if edge_attr is not None:
            edge_encoder = self.edge_encoder[self.current_edge_type_str]
            edge_encoder = edge_encoder.to(edge_attr.device)  # ✅ 修复关键

            if edge_attr.size(0) > 1:
                mean = edge_attr.mean(dim=0, keepdim=True)
                std = edge_attr.std(dim=0, keepdim=True) + 1e-6
                edge_attr = (edge_attr - mean) / std
            else:
                edge_attr = torch.zeros_like(edge_attr)

            edge_emb = edge_encoder(edge_attr).view(-1, self.heads, self.out_channels // self.heads)
            edge_emb = F.layer_norm(edge_emb, edge_emb.shape[-1:])
            edge_emb = torch.nan_to_num(edge_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            alpha += (x_j * edge_emb).sum(dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.out_channels}, heads={self.heads}, concat={self.concat})'