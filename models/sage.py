import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
from torch_geometric.data import NeighborSampler
from torch_sparse import SparseTensor


class SageConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True, root_weight=True):
        super(SageConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_l = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias_l = Parameter(torch.FloatTensor(out_features))
        self.weight_r = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias_r = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        self.root_weight = root_weight
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # self.linear = torch.nn.Linear(self.in_features, self.out_features)

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 1. / math.sqrt(self.weight_l.T.size(1))
        self.weight_l.data.uniform_(-stdv, stdv)
        self.bias_l.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_r.T.size(1))
        self.weight_r.data.uniform_(-stdv, stdv)
        self.bias_r.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, size=None):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight_l)
        else:
            support = torch.mm(input, self.weight_l)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        output = output + self.bias_l

        if self.root_weight:
            if size is not None:
                output = output + input[:size[1]] @ self.weight_r + self.bias_r
            else:
                output = output + input @ self.weight_r + self.bias_r
        else:
            output = output

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GraphSage, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(SageConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(SageConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        # TODO: do we need normalization?
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            # x_target = x[: size[1]]
            # x = self.layers[ix]((x, x_target), edge_index)
            # adj = adj.to(self.device)
            x = self.layers[ix](x, adj, size=size)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)
        
    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

