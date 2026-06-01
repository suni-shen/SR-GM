import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            print("data.is_sparse")
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
           # print("adj and support shape is {} and {}".format(adj,support))
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MMGCN(nn.Module):
    """PyG-based multimodal GCN whose ctor matches ``models.gcn.GCN``."""

    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers=2,
        dropout=0.5,
        lr=0.01,
        weight_decay=5e-4,
        with_relu=True,
        with_bias=True,
        with_bn=False,
        device=None,
    ):
        super().__init__()

        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.hid_size = nhid
        self.nlayers = nlayers

        self.lr = lr
        self.weight_decay = 0 if not with_relu else weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        if with_bn and nlayers > 1:
            warnings.warn(
                "MMGCN (PyG) ignores with_bn; batch normalization is not supported.",
                RuntimeWarning,
            )
        self.with_bn = False
        self.multi_label = None

        self.dropout = nn.Dropout(dropout)
        self._num_nodes = 0
        self.id_capacity = 0

        self.v_convs = nn.ModuleList()
        self.t_convs = nn.ModuleList()
        self._build_branches()

        self.id_embedding = nn.Parameter(torch.empty(0, self.hid_size))

    def ensure_id_capacity(self, num_nodes: int, device: Optional[torch.device] = None) -> None:
        num_nodes = int(num_nodes)
        if num_nodes <= self.id_capacity:
            return
        if device is None:
            if self.device is not None:
                device = torch.device(self.device)
            else:
                device = self.id_embedding.device
        if device is None:
            device = torch.device("cpu")
        new_embed = torch.randn(num_nodes, self.hid_size, device=device) * 0.1
        nn.init.xavier_normal_(new_embed)
        if self.id_capacity > 0 and self.id_embedding.numel() > 0:
            limit = min(self.id_capacity, num_nodes)
            new_embed[:limit] = self.id_embedding.data[:limit].to(device)
        self.id_embedding = nn.Parameter(new_embed)
        self.id_capacity = num_nodes

    def _build_branches(self):
        if self.nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        def append(container, in_dim, out_dim):
            container.append(GraphConvolution(in_dim, out_dim, with_bias=self.with_bias))

        if self.nlayers == 1:
            append(self.v_convs, self.nfeat, self.nclass)
            append(self.t_convs, self.nfeat, self.nclass)
            return

        if self.nlayers == 2:
            append(self.v_convs, self.nfeat, self.hid_size)
            append(self.v_convs, self.nfeat + self.hid_size, self.nclass)
            append(self.t_convs, self.nfeat, self.hid_size)
            append(self.t_convs, self.nfeat + self.hid_size, self.nclass)
            return

        # nlayers >= 3
        append(self.v_convs, self.nfeat, self.hid_size)
        append(self.t_convs, self.nfeat, self.hid_size)

        append(self.v_convs, self.nfeat + self.hid_size, self.hid_size)
        append(self.t_convs, self.nfeat + self.hid_size, self.hid_size)

        for _ in range(self.nlayers - 3):
            append(self.v_convs, 2 * self.hid_size, self.hid_size)
            append(self.t_convs, 2 * self.hid_size, self.hid_size)

        append(self.v_convs, 2 * self.hid_size, self.nclass)
        append(self.t_convs, 2 * self.hid_size, self.nclass)

    def _ensure_id_embedding(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        if self.id_capacity == 0:
            self.ensure_id_capacity(num_nodes, device=device)
        if num_nodes > self.id_capacity:
            raise RuntimeError(
                f"MMGCN id_embedding capacity ({self.id_capacity}) is smaller than required nodes ({num_nodes}). "
                "Call `ensure_id_capacity` with a larger value before forward passes."
            )
        self._num_nodes = max(self._num_nodes, num_nodes)
        if self.id_embedding.numel() > 0 and self.id_embedding.device != device:
            self.id_embedding.data = self.id_embedding.data.to(device)
        return self.id_embedding

    def _prepare_ids(self, x, node_indices=None):
        device = x.device
        if node_indices is not None:
            current_ids = node_indices.to(device).long()
        else:
            current_ids = torch.arange(x.size(0), device=device)
        if current_ids.numel() == 0:
            current_ids = torch.zeros(1, dtype=torch.long, device=device)
        max_id = int(current_ids.max().item()) + 1 if current_ids.numel() > 0 else 1
        embed_table = self._ensure_id_embedding(max_id, device)
        if current_ids.size(0) != x.size(0) and node_indices is None:
            current_ids = torch.arange(x.size(0), device=device)
        return current_ids, embed_table

    def _layer_forward(self, h, conv, adj, current_ids, embed_table, layer_idx, is_last):
        h_temp = conv(h, adj)
        rows = h_temp.size(0)
        target_ids = current_ids[:rows]
        if target_ids.numel() == 0:
            target_ids = torch.zeros(rows, dtype=torch.long, device=h_temp.device)
        max_id = int(target_ids.max().item()) + 1 if target_ids.numel() > 0 else 1
        embed_table = self._ensure_id_embedding(max_id, h_temp.device)
        embed = embed_table[target_ids]
        if embed.size(1) != h_temp.size(1):
            if embed.size(1) >= h_temp.size(1):
                embed = embed[:, : h_temp.size(1)]
            else:
                pad_dim = h_temp.size(1) - embed.size(1)
                embed = F.pad(embed, (0, pad_dim))
        x_hat = h_temp + embed

        if is_last or self.nlayers == 1:
            return x_hat, target_ids, embed_table

        if layer_idx == 0:
            prefix = h[:rows]
        else:
            prefix = h[:rows, -self.hid_size:]

        out = torch.cat([prefix, x_hat], dim=1)

        if self.with_relu:
            out = F.leaky_relu(out)
        out = self.dropout(out)

        return out, target_ids, embed_table

    def forward(self, x, edge_index, batch=None, node_indices=None):
        del batch  # batch assignment is unused in this formulation
        x = F.normalize(x, p=2, dim=1)
        current_ids, embed_table = self._prepare_ids(x, node_indices=node_indices)

        h_v = x
        h_t = x

        for idx in range(len(self.v_convs)):
            is_last = idx == len(self.v_convs) - 1
            h_v, new_ids, embed_table = self._layer_forward(
                h_v, self.v_convs[idx], edge_index, current_ids, embed_table, idx, is_last
            )
            h_t, _, embed_table = self._layer_forward(
                h_t, self.t_convs[idx], edge_index, current_ids, embed_table, idx, is_last
            )
            current_ids = new_ids

        logits = (h_v + h_t) / 2
        if self.multi_label:
            return torch.sigmoid(logits)
        return F.log_softmax(logits, dim=1)

    def forward_sampler(self, x, adjs, node_indices=None):
        x = F.normalize(x, p=2, dim=1)
        current_ids, embed_table = self._prepare_ids(x, node_indices=node_indices)

        h_v = x
        h_t = x

        for idx, adj_pack in enumerate(adjs):
            if idx >= len(self.v_convs):
                break
            if isinstance(adj_pack, (tuple, list)):
                adj = adj_pack[0]
            else:
                adj = adj_pack
            adj = adj.to(x.device)
            is_last = idx == len(self.v_convs) - 1 or idx == len(adjs) - 1
            h_v, new_ids, embed_table = self._layer_forward(
                h_v, self.v_convs[idx], adj, current_ids, embed_table, idx, is_last
            )
            h_t, _, embed_table = self._layer_forward(
                h_t, self.t_convs[idx], adj, current_ids, embed_table, idx, is_last
            )
            current_ids = new_ids

        logits = (h_v + h_t) / 2
        if self.multi_label:
            return torch.sigmoid(logits)
        return F.log_softmax(logits, dim=1)

    def initialize(self):
        for conv in self.v_convs:
            conv.reset_parameters()
        for conv in self.t_convs:
            conv.reset_parameters()
        if self.id_capacity > 0:
            device = self.id_embedding.device if self.id_embedding.numel() > 0 else (
                torch.device(self.device) if self.device is not None else torch.device("cpu")
            )
            new_embed = torch.randn(self.id_capacity, self.hid_size, device=device) * 0.1
            nn.init.xavier_normal_(new_embed)
            self.id_embedding = nn.Parameter(new_embed)
        else:
            self.id_embedding = nn.Parameter(torch.empty(0, self.hid_size))
