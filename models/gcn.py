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


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

       # assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

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
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
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

    def fit(self, features, adj,data, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, **kwargs):

        if initialize:
            self.initialize()

        # features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features
        self.feature_full=data.feat_full
        self.adj_full=data.adj_full
        self.lable_full=data.labels_full


        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        self._train_with_val_induct(labels,  train_iters, verbose)
       # if idx_val is not None:
       #     self._train_with_val2(labels, idx_train, idx_val, train_iters, verbose)
       # else:
       #     self._train_without_val2(labels, idx_train, train_iters, verbose)
    def _train_with_val_induct(self, labels, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output , labels )
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_without_val2(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val(labels, data, train_iters, verbose)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        #feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val =data.labels_val #torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
            '''
            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.val_mask], labels_val)
                    acc_val = utils.accuracy(output[data.val_mask], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
            '''
        #if verbose:
       #     print('=== picking the best model according to the performance on validation ===')
       # self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict(self.feature_full,self.adj_full)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.lable_full[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.lable_full[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
   

    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)


    def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


class MMGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(MMGCN, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.hid_size = nhid
        self.nlayers = nlayers
        self._num_nodes = 0

        self.v_layers = nn.ModuleList([])
        self.t_layers = nn.ModuleList([])

        if nlayers == 1:
            self.v_layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
            self.t_layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.v_bns = torch.nn.ModuleList()
                self.t_bns = torch.nn.ModuleList()
            self.v_layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            self.t_layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            if with_bn:
                self.v_bns.append(nn.BatchNorm1d(nhid))
                self.t_bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-2):
                self.v_layers.append(GraphConvolution(2 * nhid, nhid, with_bias=with_bias))
                self.t_layers.append(GraphConvolution(2 * nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.v_bns.append(nn.BatchNorm1d(nhid))
                    self.t_bns.append(nn.BatchNorm1d(nhid))
            self.v_layers.append(GraphConvolution(2 * nhid, nclass, with_bias=with_bias))
            self.t_layers.append(GraphConvolution(2 * nhid, nclass, with_bias=with_bias))

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
        self.id_embedding = nn.Parameter(torch.empty(0, nhid))

    def _ensure_id_embedding(self, num_nodes, device):
        if num_nodes <= 0:
            num_nodes = 1
        if self.id_embedding.numel() == 0:
            id_embed = torch.randn(num_nodes, self.hid_size, device=device) * 0.1
            nn.init.xavier_normal_(id_embed)
            self.id_embedding = nn.Parameter(id_embed)
        else:
            current = self.id_embedding
            if current.device != device:
                current = current.to(device)
            if current.size(0) < num_nodes:
                extra = torch.randn(num_nodes - current.size(0), self.hid_size, device=device) * 0.1
                nn.init.xavier_normal_(extra)
                current = torch.cat([current, extra], dim=0)
                self.id_embedding = nn.Parameter(current)
            elif current is not self.id_embedding:
                self.id_embedding = nn.Parameter(current)
        return self.id_embedding

    def _adj_num_rows(self, adj):
        if isinstance(adj, torch_sparse.SparseTensor):
            return adj.size(0)
        elif torch.is_tensor(adj):
            return adj.size(0)
        else:
            raise TypeError(f"Unsupported adjacency type: {type(adj)}")

    def _branch_forward(self, layers, bns, x, adj, id_embed_full):
        h = x
        for idx, layer in enumerate(layers):
            h_temp = layer(h, adj)
            rows = h_temp.size(0)

            if id_embed_full.size(0) < rows:
                id_embed_full = self._ensure_id_embedding(rows, h_temp.device)
            embed = id_embed_full[:rows]
            embed = embed.to(h_temp.device)

            if idx != len(layers) - 1:
                if bns is not None:
                    h_temp = bns[idx](h_temp)
                if embed.shape[1] == h_temp.shape[1]:
                    x_hat = h_temp + embed
                else:
                    x_hat = h_temp
                h = torch.cat([h_temp, x_hat], dim=1)
                if self.with_relu:
                    h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                if embed.shape[1] == h_temp.shape[1]:
                    h_temp = h_temp + embed
                return h_temp
        return h

    def forward(self, x, adj):
        x = F.normalize(x, p=2, dim=1)
        device = x.device
        target_nodes = self._adj_num_rows(adj)
        if target_nodes > self._num_nodes:
            self._num_nodes = target_nodes
        id_embed = self._ensure_id_embedding(target_nodes, device)

        if self.nlayers == 1:
            v_out = self.v_layers[0](x, adj)
            t_out = self.t_layers[0](x, adj)
        else:
            bns_v = self.v_bns if self.with_bn else None
            bns_t = self.t_bns if self.with_bn else None
            v_out = self._branch_forward(self.v_layers, bns_v, x, adj, id_embed)
            id_embed = self.id_embedding  # refresh in case resized
            t_out = self._branch_forward(self.t_layers, bns_t, x, adj, id_embed)

        logits = (v_out + t_out) / 2

        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.log_softmax(logits, dim=1)

    def forward_sampler(self, x, adjs):
        adjs = list(adjs)
        x = F.normalize(x, p=2, dim=1)
        device = x.device
        if x.size(0) > self._num_nodes:
            self._num_nodes = x.size(0)

        h_v = x
        h_t = x
        bns_v = getattr(self, "v_bns", None) if self.with_bn and self.nlayers > 1 else None
        bns_t = getattr(self, "t_bns", None) if self.with_bn and self.nlayers > 1 else None
        v_out = None
        t_out = None

        for ix, adj_pack in enumerate(adjs):
            if isinstance(adj_pack, tuple):
                if len(adj_pack) == 3:
                    adj, _, size = adj_pack
                elif len(adj_pack) == 2:
                    adj, size = adj_pack
                else:
                    adj = adj_pack[0]
                    size = (h_v.size(0), h_v.size(0))
            else:
                adj = adj_pack
                size = (h_v.size(0), h_v.size(0))

            if isinstance(size, torch.Size):
                size = tuple(size)
            v_temp = self.v_layers[ix](h_v, adj)
            t_temp = self.t_layers[ix](h_t, adj)

            rows = v_temp.size(0)

            if isinstance(size, (tuple, list)) and len(size) >= 2:
                out_nodes = size[1]
            else:
                out_nodes = rows

            max_nodes = max(self._num_nodes, rows, out_nodes)
            if max_nodes > self._num_nodes:
                self._num_nodes = max_nodes
            id_embed_full = self._ensure_id_embedding(self._num_nodes, device)
            id_embed_tgt = id_embed_full[:rows]

            if ix != len(self.v_layers) - 1:
                if bns_v is not None:
                    v_temp = bns_v[ix](v_temp)
                    t_temp = bns_t[ix](t_temp)
                if self.with_relu:
                    v_temp = F.relu(v_temp)
                    t_temp = F.relu(t_temp)
                if id_embed_tgt.shape[1] == v_temp.shape[1]:
                    v_hat = v_temp + id_embed_tgt[:v_temp.size(0)]
                    t_hat = t_temp + id_embed_tgt[:t_temp.size(0)]
                else:
                    v_hat = v_temp
                    t_hat = t_temp
                h_v = torch.cat([v_temp, v_hat], dim=1)
                h_t = torch.cat([t_temp, t_hat], dim=1)
                h_v = F.dropout(h_v, self.dropout, training=self.training)
                h_t = F.dropout(h_t, self.dropout, training=self.training)
            else:
                if id_embed_tgt.shape[1] == v_temp.shape[1]:
                    v_temp = v_temp + id_embed_tgt[:v_temp.size(0)]
                    t_temp = t_temp + id_embed_tgt[:t_temp.size(0)]
                v_out = v_temp
                t_out = t_temp

        logits = (v_out + t_out) / 2

        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.log_softmax(logits, dim=1)

    def initialize(self):
        for layer in self.v_layers:
            layer.reset_parameters()
        for layer in self.t_layers:
            layer.reset_parameters()
        if self.with_bn and self.nlayers > 1:
            for bn in self.v_bns:
                bn.reset_parameters()
            for bn in self.t_bns:
                bn.reset_parameters()
        self.id_embedding = nn.Parameter(torch.empty(0, self.hid_size))
