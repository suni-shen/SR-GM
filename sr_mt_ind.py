import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from torch.nn import Parameter
from torch_geometric.data import Data
import torch_geometric.transforms as T
from tqdm import tqdm
import deeprobust.graph.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from nc_dataset_pyg import NodeClassificationDataset
from models import KCenter, Random, Herding
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from utils import seed_everything, match_loss, regularization, row_normalize_tensor
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch.nn.parameter import Parameter
from itertools import product
from models import GCN, GraphSage, MLP, MMGCN, GAT
from torch_geometric.data import NeighborSampler
import torch_geometric as pyg
import  torch_geometric.utils as tg_utils
from copy import deepcopy

import torch.optim as optim
from copy import deepcopy
from torch_geometric.utils import subgraph

 
PROJETC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), './')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)
os.makedirs(f"{PROJETC_DIR}/saved-ours/srgm/", exist_ok=True)


def sparse_submatrix(adj, row_indices, col_indices=None):
    if col_indices is None:
        col_indices = row_indices
    
    if adj.is_sparse:
        adj_dense = adj.to_dense()
        result = adj_dense[row_indices][:, col_indices]
        
        return result.to_sparse()
    else:
        return adj[row_indices][:, col_indices]


def adj_dense_to_sp(adj_sp_dense):
    tmp_coo = sp.coo_matrix(adj_sp_dense)
    values = tmp_coo.data
    indices = np.vstack((tmp_coo.row,tmp_coo.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    adj_sp=torch.sparse_coo_tensor(i,v,tmp_coo.shape)
    return adj_sp

def get_loops(cfg):
    return 20, 10

def adj_to_edge_index( adj_matrix, threshold=0.5):
        adj_binary = (adj_matrix > threshold).float()
        edge_indices = torch.nonzero(adj_binary, as_tuple=False).t()
        return edge_indices
    
def edge_index_to_adj(edge_indices, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_indices.max().item() + 1

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    adj_matrix[edge_indices[0], edge_indices[1]] = 1.0
    
    return adj_matrix

def create_pyg_data(edges, features, labels, train_mask, val_mask, test_mask,num_classes):
    """Convert DGL-style data to PyG Data object"""
    # Create edge_index (2 x num_edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Add reverse edges to make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Remove duplicates and self-loops
    edge_index = torch.unique(edge_index, dim=1)
    
    adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])
    
    adj_sparse = pyg.utils.to_torch_sparse_tensor(edge_index, 
                                             size=(features.size(0), features.size(0)))

    adj_scipy = sp.csr_matrix((adj_sparse.values().numpy(),
                          (adj_sparse.indices()[0].numpy(), 
                           adj_sparse.indices()[1].numpy())),
                         shape=adj_sparse.shape)

    adj_de_train = adj_scipy[np.ix_(train_mask, train_mask)]
    adj_de_test = adj_scipy[np.ix_(test_mask, test_mask)]

    adj_full=adj_dense_to_sp(adj_scipy)
    adj_sp_train=adj_dense_to_sp(adj_de_train)
    adj_sp_test=adj_dense_to_sp(adj_de_test)

    idx_train_full=torch.ones(features.size(0), dtype=torch.bool)

    data = Data(
        feat_full=features,
        adj_full=adj_full,
        labels_full=labels,
        x=features[train_mask],
        feat_test=features[test_mask],
        edge_index=edge_index,
        labels=labels[train_mask],
        idx_train_full=idx_train_full,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=features[train_mask].size(0),
        labels_train=labels[train_mask],
        labels_val=labels[val_mask],
        labels_test=labels[test_mask],
        adj_train=adj_sp_train,
        adj_test=adj_sp_test,
        nclass=num_classes 
    )
    
    return data

class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.reduction_rate==0.01:
               nhid = 128
           nlayers = 3

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
            edge_index = self.edge_index
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]],
                        x[edge_index[1][idx]]], axis=1)
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)
            edge_embed = torch.cat(edge_embed)
        else:
            edge_index = self.edge_index
            edge_embed = torch.cat([x[edge_index[0]],
                    x[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

class MMGraphCompressor:
    """MMGraph Compressor for graph condensation"""
    def __init__(self, cfg: DictConfig ):
        self.data_path =cfg.data_path
        self.reduction_rate = cfg.reduction_rate
        self.device = cfg.device
        self.cfg=cfg
        self.data, self.model=self.load_data_and_model(self.cfg)
        
        # Set up compression parameters
        self.setup_compression()
   
    def load_data_and_model(self,cfg ):

         # save configs
        if not os.path.isfile("config.yaml"):
            OmegaConf.save(config=cfg, f=os.path.join("config.yaml"))

        # load and preprocess dataset
        device = torch.device('cpu' if cfg.mode == 'cpu' else 'cuda')

        # load and preprocess dataset
        log.info("Loading data")

        data_path = cfg.data_path
        dataset_name = cfg.dataset
        feat_name = cfg.feat #'clip' #'t5vit' imagebind
        verbose = True
        
        log.info(f"Dataset: {dataset_name}, Feature: {feat_name}")
        log.info(f"lambda: {cfg.lambad}, reduction_rate: {cfg.reduction_rate}")

        dataset = NodeClassificationDataset(
            root=os.path.join(data_path, dataset_name),
            feat_name=feat_name,
            verbose=verbose,
            device='cpu'  # Load on CPU first
        )

        edges = dataset.edge
        labels = dataset.labels
        
        self.edge_index=edges
        
        self.labels=labels
        # Load features based on configuration
        if cfg.feat == 'clip':
            features = torch.load(os.path.join(data_path, dataset_name, 'clip_feat.pt'))
        elif cfg.feat == 'imagebind':
            features = torch.load(os.path.join(data_path, dataset_name, 'imagebind_feat.pt'))
        elif cfg.feat == 'dino':
            features = torch.load(os.path.join(data_path, dataset_name, 't5dino_feat.pt'))
        else: 
            features = torch.load(os.path.join(data_path, dataset_name, 't5vit_feat.pt'))
        
        log.info("features shape is {}".format(features.shape))
         
        self.text_features = features[:, :int(features.shape[1] / 2)]
        self.image_features = features[:, int(features.shape[1] / 2):]
        self.features = features 
        # Create masks
        self.train_mask = torch.zeros(features.size(0), dtype=torch.bool)
        self.val_mask = torch.zeros(features.size(0), dtype=torch.bool)
        self.test_mask = torch.zeros(features.size(0), dtype=torch.bool)
        
        
        node_split = dataset.node_split
        self.train_mask[node_split['train_idx']] = True
        self.val_mask[node_split['val_idx']] = True
        self.test_mask[node_split['test_idx']] = True
        self.class_dict = None
        self.class_dict2 = None
        self.samplers = None
        self.labels_train=labels[self.train_mask]
        self.labels_val=labels[self.val_mask]
        self.labels_test=labels[self.test_mask]
        if dataset_name in ['books_nc', 'books_lp'] :
            
            log.info("using books")
            self.num_classes = len(torch.unique(labels))-1
        
        else:
            self.num_classes = 12
            
        # Create PyG data object
        data = create_pyg_data(edges, self.features, labels, self.train_mask, self.val_mask, self.test_mask,self.num_classes)
        self.edge_index=data.edge_index
        self.edge_index_train  = subgraph(
            self.train_mask, 
            self.edge_index, 
            relabel_nodes=True,        
            num_nodes=features[self.train_mask].size(0)
            )[0]
       
        log.info("self.edge_index_train is {}".format(self.edge_index_train))
        data = data.to(device)
        
        
        in_size = data.x.shape[1]
        out_size = self.num_classes
        log.info("device is {}".format(device))
        log.info("data is {}".format(data))

        model=eval(self.cfg.model_name)(nfeat=in_size, nhid=self.cfg.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        return data,model
    
    def test_with_val(self, verbose=False):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.syn_features.detach(), \
                                self.pge, self.labels_syn
        
        dropout = 0.5  
        model = eval(self.cfg.model_name)(nfeat=feat_syn.shape[1], nhid=self.cfg.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,with_bn=True,
                    nclass=data.nclass, device=device).to(device)

        if hasattr(model, "ensure_id_capacity"):
            total_nodes = data.feat_full.size(0) if hasattr(data, "feat_full") else data.x.size(0)
            model.ensure_id_capacity(total_nodes, device=device)

        adj_syn = pge.inference(feat_syn)
    
        adj_norm = utils.normalize_adj_tensor(adj_syn, sparse=True)
        adj_test_norm = utils.normalize_adj_tensor(data.adj_test, sparse=True)

        loss = F.nll_loss

        if verbose:
            print('=== training gcn model ===')
        
        lr = self.cfg.lr_test
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        best_acc_val = 0
        train_iters=300
        for i in range(train_iters):
            if i == train_iters // 2:
                optimizer = optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=5e-4)

            model.train()
            optimizer.zero_grad()
            output = model.forward(feat_syn, adj_norm)
            loss_train = loss(output, labels_syn)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                model.eval()
                output = model.forward(data.feat_test, adj_test_norm)

                acc_val = utils.accuracy(output, data.labels_test)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(model.state_dict())
                    if verbose:
                        print('Epoch {}, best_acc_val: {}'.format(i, best_acc_val))

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        model.load_state_dict(weights)
        
        model.eval()
        labels_test =  data.labels_test
        
        output = model.forward(data.feat_test, adj_test_norm)
        
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))

        return res

    def test_from_saved(self, feat_path=None, adj_path=None, runs_eval=None, verbose=True):
        """Test using saved synthetic features/adjacency.
        Args:
            feat_path: path to saved feat tensor (.pt). If None, uses default naming.
            adj_path: path to saved adj tensor (.pt). If None, uses default naming.
            runs_eval: number of repeated evaluations. Defaults to self.cfg.runs_eval.
            verbose: whether to print training progress.
        """
        data = self.data
        device = self.device

        base_dir = os.path.join(PROJETC_DIR, 'saved-ours', 'srgm')
        if feat_path is None or adj_path is None:
            if self.cfg.ablation:
                default_adj = os.path.join(
                    base_dir,
                    f'ablation_adj_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt',
                )
                default_feat = os.path.join(
                    base_dir,
                    f'ablation_feat_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt',
                )
            else:
                default_adj = os.path.join(
                    base_dir,
                    f'adj_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt',
                )
                default_feat = os.path.join(
                    base_dir,
                    f'feat_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt',
                )
            if feat_path is None:
                feat_path = default_feat
            if adj_path is None:
                adj_path = default_adj

        if not os.path.isfile(feat_path):
            raise FileNotFoundError(f"feat_path not found: {feat_path}")
        if not os.path.isfile(adj_path):
            raise FileNotFoundError(f"adj_path not found: {adj_path}")

        feat_syn = torch.load(feat_path, map_location=device)
        adj_syn = torch.load(adj_path, map_location=device)

        # labels_syn should already exist from setup_compression(); fallback just in case.
        labels_syn = getattr(self, 'labels_syn', None)
        if labels_syn is None:
            labels_syn = torch.LongTensor(self.generate_labels_syn(self.data)).to(device)

        if runs_eval is None:
            runs_eval = int(getattr(self.cfg, 'runs_eval', 3))

        all_res = []
        for _ in range(runs_eval):
            res = []

            dropout = 0.5
            model = eval(self.cfg.model_name)(nfeat=feat_syn.shape[1], nhid=self.cfg.hidden, dropout=dropout,
                        weight_decay=5e-4, nlayers=2,with_bn=True,
                        nclass=data.nclass, device=device).to(device)

            if hasattr(model, "ensure_id_capacity"):
                total_nodes = data.feat_full.size(0) if hasattr(data, "feat_full") else data.x.size(0)
                model.ensure_id_capacity(total_nodes, device=device)

            adj_norm = utils.normalize_adj_tensor(adj_syn, sparse=True)
            adj_test_norm = utils.normalize_adj_tensor(data.adj_test, sparse=True)

            loss = F.nll_loss

            if verbose:
                print('=== training gcn model ===')

            lr = self.cfg.lr_test
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            best_acc_val = 0
            train_iters=300
            weights = None
            for i in range(train_iters):
                if i == train_iters // 2:
                    optimizer = optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=5e-4)

                model.train()
                optimizer.zero_grad()
                output = model.forward(feat_syn, adj_norm)
                loss_train = loss(output, labels_syn)
                loss_train.backward()
                optimizer.step()

                if verbose and i % 50 == 0:
                    print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

                with torch.no_grad():
                    model.eval()
                    output = model.forward(data.feat_test, adj_test_norm)
                    acc_val = utils.accuracy(output, data.labels_test)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        self.output = output
                        weights = deepcopy(model.state_dict())
                        if verbose:
                            print('Epoch {}, best_acc_val: {}'.format(i, best_acc_val))

            if weights is None:
                weights = deepcopy(model.state_dict())

            if verbose:
                print('=== picking the best model according to the performance on validation ===')

            model.load_state_dict(weights)
            model.eval()

            labels_test =  data.labels_test
            output = model.forward(data.feat_test, adj_test_norm)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            print("Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))

            all_res.append(res)

        all_res = np.array(all_res)
        res_mean = all_res.mean()
        res_std = all_res.std()
        print(f"[Saved] Mean accuracy: {res_mean:.4f} ± {res_std:.4f}")
        return all_res
    
    def setup_compression(self):
        
        n_synthetic = int(len(self.labels_train) * self.reduction_rate)
        self.n_synthetic = n_synthetic
        
        # Obtain feature dime
        self.text_dim = self.text_features.shape[1]
        self.image_dim = self.image_features.shape[1]
        self.feat_dim=self.features.shape[1]
        
        # Initialize synthetic graph
        self.syn_features = Parameter(torch.FloatTensor(n_synthetic, self.feat_dim).to(self.device))
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(self.data)).to(self.device)
        self.pge = PGE(nfeat=self.feat_dim , nnodes=n_synthetic, device=self.device,args=self.cfg).to(self.device)
        
        if self.cfg.debug:
            print(f"Compression setup: {len(self.labels_train)} -> {n_synthetic} nodes")
            print(f"Text dim: {self.text_dim}, Image dim: {self.image_dim}")
        log.info(f"Compression setup: {len(self.labels_train)} -> {n_synthetic} nodes")

        self.optimizer_feat = torch.optim.Adam([self.syn_features], lr=self.cfg.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=self.cfg.lr_adj)
    
    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train.cpu().numpy())
       
        num_class_dict = {}
        n = len(data.labels_train)
       
        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        log.info("sorted counter is {}".format(sorted_counter))
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.cfg.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.cfg.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        log.info("labels_syn is {}".format(len(labels_syn)))
        return labels_syn
    
    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.num_classes):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [20, 10]

        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.num_classes):
                if transductive:
                    idx_train = np.array(self.train_mask)
                    idx = idx_train[self.labels_train == i]
                else:
                   
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx
      
        if self.samplers is None:
            self.samplers = []
            for i in range(self.num_classes):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    self.samplers.append([])
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
      
        out = self.samplers[c].sample(batch)
        return out

    def get_sub_adj_feat(self, features):
        data = self.data
        idx_selected = []
        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = self.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        from sklearn.metrics.pairwise import cosine_similarity
        
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn
    
    def get_laplacian(self,adj):
        """
        L = I - D^(-1/2) * A * D^(-1/2)
        """

        adj_self_loop = adj + torch.eye(adj.shape[0], device=adj.device)
        deg = torch.sum(adj_self_loop, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L_norm = torch.eye(adj.shape[0], device=adj.device) - D_inv_sqrt @ adj_self_loop @ D_inv_sqrt
        return L_norm

    def train_compression(self):
        """ Train the synthetic graph"""
        log.info("Starting compression training...")
        verbose=True

        data=self.data
        feat_syn, pge, labels_syn = self.syn_features, self.pge, self.labels_syn
        features, adj, labels = data.x, data.adj_train, data.labels_train
         
        syn_class_indices = self.syn_class_indices

        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        feat_syn.data.copy_(feat_sub)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()


        outer_loop, inner_loop = get_loops(self.cfg)
        log.info(f"Outer loop: {outer_loop}, Inner loop: {inner_loop}")
        loss_avg = 0

        best_res_mean = 0
        best_res_std = 0
        for it in range(self.cfg.epochs+1):
             
            model = eval(self.cfg.model_name)(nfeat=data.x.shape[1], nhid=self.cfg.hidden,with_bn=False,
                                nclass=data.nclass, dropout=self.cfg.dropout, nlayers=self.cfg.nlayers,
                                device=self.device).to(self.device)

            if hasattr(model, "ensure_id_capacity"):
                total_nodes = data.feat_full.size(0) if hasattr(data, "feat_full") else data.x.size(0)
                model.ensure_id_capacity(total_nodes, device=self.device)


            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=self.cfg.lr_model)

            model.train()

            for ol in range(outer_loop):
               
                adj_syn = pge(self.syn_features)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn
                
                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            print("BatchNorm")
                            module.eval() # fix mu and sigma of every BatchNorm layer
                
                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue
                    
                    batch_size, n_id, adjs = self.retrieve_class_sampler(
                            c, adj, transductive=False, args=self.cfg)
                    
                    adjs = [adj.to(self.device) for adj in adjs]
                    if self.cfg.model_name == "MLP":
                        output = model.forward_sampler(features[n_id[:batch_size]], adjs)
                    else:
                        if hasattr(model, "ensure_id_capacity"):
                            output = model.forward_sampler(features[n_id], adjs, node_indices=n_id)
                        else:
                            output = model.forward_sampler(features[n_id], adjs)
                    
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])

                    gw_real = torch.autograd.grad(loss_real, model_parameters, retain_graph=True)
                    gw_real = [grad.detach().clone() for grad in gw_real]

                    ind = syn_class_indices[c]
                    if self.cfg.model_name in ['GCN', 'MMGCN']:
                        if self.cfg.nlayers == 1:
                            adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                        else:
                            adj_syn_norm_list = [adj_syn_norm]*(self.cfg.nlayers-1) + \
                                    [adj_syn_norm[ind[0]: ind[1]]]
                        output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                        
                        loss_syn = F.nll_loss(
                                output_syn,
                                labels_syn[ind[0]: ind[1]])
                    else:
                        output_syn = model.forward(feat_syn, adj_syn_norm)
                    
                        loss_syn = F.nll_loss(
                                output_syn[ind[0]: ind[1]],
                                labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(
                        loss_syn, model_parameters, create_graph=True)
                    
                    coeff = self.num_class_dict[c] / self.n_synthetic
                    
                    loss += coeff * match_loss(gw_syn, gw_real, self.cfg, device=self.device)

                loss_avg += loss.item()

                # --- SR-GM ---
                output_syn_sr = model.forward(feat_syn, adj_syn_norm)
                loss_syn_sr = F.nll_loss(output_syn_sr,labels_syn)
                g_feat_syn_raw = torch.autograd.grad(loss_syn_sr, feat_syn, retain_graph=True)[0]

                L_syn = self.get_laplacian(adj_syn)
                
                if self.cfg.ablation:
                    loss_reg = torch.trace(g_feat_syn_raw.T @ L_syn @ g_feat_syn_raw)
                else:
                    # --- Component I: Gradient Decoupling ---
                    g_text = g_feat_syn_raw[:, :self.cfg.d_text]
                    g_image = g_feat_syn_raw[:, self.cfg.d_text:]
                    
                    dot_products = torch.sum(g_text * g_image, dim=1)

                    conflict_mask = (dot_products < 0).view(-1, 1)

                    norm_sq_text = torch.sum(g_text**2, dim=1, keepdim=True) + 1e-8
                    norm_sq_image = torch.sum(g_image**2, dim=1, keepdim=True) + 1e-8

                    proj_factor_on_image = (dot_products / norm_sq_image.squeeze()).view(-1, 1)
                    proj_factor_on_text = (dot_products / norm_sq_text.squeeze()).view(-1, 1)

                    g_text_proj = g_text - proj_factor_on_image * g_image
                    g_image_proj = g_image - proj_factor_on_text * g_text

                    final_g_text = torch.where(conflict_mask, g_text_proj, g_text)
                    final_g_image = torch.where(conflict_mask, g_image_proj, g_image)

                    decoupled_grads = torch.cat((final_g_text, final_g_image), dim=1)

                    # --- Component II: Structural Regularization (Damping) ---
                    # Calculate the Dirichlet energy of the gradient field: tr(G'^T * L * G')
                    loss_reg = torch.trace(decoupled_grads.T @ L_syn @ decoupled_grads)
                    
                loss = loss + self.cfg.lambad * loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if ol == outer_loop - 1:
                    break
                
                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    
                    optimizer_model.step() # update gnn param
            
            loss_avg /= (data.nclass*outer_loop)
            if (it + 1) % 20 == 0:
                log.info('Epoch {}, loss_avg: {}'.format(it, loss_avg))
            
            if verbose and (it+1) % 20 == 0:
                res = []
                for i in range(self.cfg.runs_eval):
                    res.append(self.test_with_val())

                res = np.array(res)
                res_mean = res.mean()
                res_std = res.std()
                if res_mean > best_res_mean:
                    best_res_mean = res_mean
                    best_res_std = res_std
                
                    best_feat_syn, pge = self.syn_features.detach(), self.pge
                    best_adj_syn = pge.inference(best_feat_syn).detach()
                    if self.cfg.save:
                        if self.cfg.ablation:
                            torch.save(best_adj_syn, f'{PROJETC_DIR}/saved-ours/srgm/ablation_adj_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt')
                            torch.save(best_feat_syn, f'{PROJETC_DIR}/saved-ours/srgm/ablation_feat_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt')
                        else:
                            torch.save(best_adj_syn, f'{PROJETC_DIR}/saved-ours/srgm/adj_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt')
                            torch.save(best_feat_syn, f'{PROJETC_DIR}/saved-ours/srgm/feat_{self.cfg.dataset}_{self.cfg.reduction_rate}_{self.cfg.model_name}_{self.cfg.feat}.pt')
        
                log.info(f'Mean accuracy: {repr([res.mean(), res.std()])}')
                log.info(f'Best mean accuracy: {repr([best_res_mean, best_res_std])}')
                

        
@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    cfg.data_path = os.path.join(PROJETC_DIR, cfg.data_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(cfg.seed)
    
    log.info(f"Using device: {device}")

    log.info(f"{'='*50}")
    
    compressor = MMGraphCompressor(cfg)
    log.info(f"{'='*50}")
    
    if cfg.training:
        compressor.train_compression()
    else:
        compressor.test_from_saved(verbose=True)


if __name__ == "__main__":
    main()


