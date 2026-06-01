
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None,
                 hidden_heads=1, output_heads=1):

        super(GAT, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.hidden_heads = hidden_heads
        self.output_heads = output_heads

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(
                GATv2Conv(
                    nfeat,
                    nclass,
                    heads=output_heads,
                    concat=False,
                    dropout=dropout,
                    bias=with_bias,
                    add_self_loops=False,
                    edge_dim=1,
                )
            )
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid * hidden_heads))

            in_channels = nfeat
            self.layers.append(
                GATv2Conv(
                    in_channels,
                    nhid,
                    heads=hidden_heads,
                    concat=True,
                    dropout=dropout,
                    bias=with_bias,
                    add_self_loops=False,
                    edge_dim=1,
                )
            )
            out_channels = nhid * hidden_heads

            for _ in range(nlayers - 2):
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(out_channels))
                self.layers.append(
                    GATv2Conv(
                        out_channels,
                        nhid,
                        heads=hidden_heads,
                        concat=True,
                        dropout=dropout,
                        bias=with_bias,
                        add_self_loops=False,
                        edge_dim=1,
                    )
                )
                out_channels = nhid * hidden_heads

            self.layers.append(
                GATv2Conv(
                    out_channels,
                    nclass,
                    heads=output_heads,
                    concat=False,
                    dropout=dropout,
                    bias=with_bias,
                    add_self_loops=False,
                    edge_dim=1,
                )
            )

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = 0 if not with_relu else weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    @staticmethod
    def _prepare_adj(adj, device):
        if isinstance(adj, SparseTensor):
            row, col, value = adj.coo()
            edge_index = torch.stack([col, row], dim=0)
            edge_weight = value
        elif torch.is_tensor(adj):
            if adj.is_sparse:
                adj = adj.coalesce()
                indices = adj.indices()
                values = adj.values()
                edge_index = torch.stack([indices[1], indices[0]], dim=0)
                edge_weight = values
            else:
                coords = adj.nonzero(as_tuple=False)
                edge_index = torch.stack([coords[:, 1], coords[:, 0]], dim=0)
                edge_weight = adj[coords[:, 0], coords[:, 1]]
        else:
            raise TypeError(f"Unsupported adjacency type: {type(adj)}")

        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        return edge_index, edge_weight

    def _apply_activation(self, x, idx):
        if idx != len(self.layers) - 1:
            if self.with_bn:
                x = self.bns[idx](x)
            if self.with_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x, adj):
        device = x.device
        edge_index, edge_weight = self._prepare_adj(adj, device)

        for ix, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_weight)
            x = self._apply_activation(x, ix)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        adjs = list(adjs)
        if len(adjs) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} adjacency blocks, got {len(adjs)}")
        for ix, (adj, _, size) in enumerate(adjs):
            edge_index, edge_weight = self._prepare_adj(adj, x.device)
            x_target = x[:size[1]]
            x = self.layers[ix]((x, x_target), edge_index, edge_attr=edge_weight)
            x = self._apply_activation(x, ix)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
