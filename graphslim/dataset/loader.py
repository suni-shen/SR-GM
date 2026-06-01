import json
import os.path as osp
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import torch_geometric as pyg

from graphslim.dataset.convertor import ei2csr, csr2ei, from_dgl
from graphslim.dataset.utils import splits
from graphslim.utils import index_to_mask, to_tensor


def get_dataset(name='ele-fashion', args=None):
    # Create a dictionary that maps standard names to normalized names
    standard_names = ['nc_ar_sport', 'nc_ar_cloth' , 'ar_sport', 'ar_cloth', 'ele-fashion', 'books_nc']

    if name in standard_names:
        if name in ['ele-fashion', 'books_nc']:
            data_path = args.data_path
            dataset_name = name
            feat_name = args.feat_name #'clip' #'t5vit' # 'imagebind'
            dataset = NodeClassificationDataset(
                root=os.path.join(data_path, dataset_name),
                feat_name=feat_name,
                verbose=True,
                device='cpu'  # Load on CPU first
            )
            
        elif name in ['nc_ar_sport', 'nc_ar_cloth']:
            data_path = args.data_path
            dataset = torch.load(os.path.join(data_path, f'{name}_{args.feat_name}.pt'))
            dataset.num_classes = 2
            dataset.idx_train = dataset.train_mask.nonzero().view(-1)
            dataset.idx_val = dataset.val_mask.nonzero().view(-1)
            dataset.idx_test = dataset.test_mask.nonzero().view(-1)

    else:
        raise ValueError("Dataset name not recognized.")

    try:
        data = dataset[0]

    except:
        data = dataset
    
    else:
        data = splits(data, args.split)

    data = TransAndInd(data, name, args.pre_norm)

    try:
        data.nclass = dataset.num_classes
    except:
        data.nclass = data.num_classes

    print("train nodes num:", sum(data.train_mask).item())
    print("val nodes num:", sum(data.val_mask).item())
    print("test nodes num:", sum(data.test_mask).item())
    print("total nodes num:", data.x.shape[0])
    return data


class TransAndInd:

    def __init__(self, data, dataset, norm=True):
        self.class_dict = None  # sample the training data per class when initializing synthetic graph
        self.samplers = None
        self.class_dict2 = None  # sample from the same class when training
        self.sparse_adj = None
        self.adj_full = None
        self.feat_full = None
        self.labels_full = None
        self.num_nodes = data.num_nodes
        self.train_mask, self.val_mask, self.test_mask = data.train_mask, data.val_mask, data.test_mask
        self.pyg_saint(data)
        if dataset in ["nc_ar_sport", "nc_ar_cloth"]:
            self.edge_index = to_undirected(self.edge_index, self.num_nodes)
            self.feat_full = data.x

        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test

        self.adj_train = self.adj_full[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj_full[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj_full[np.ix_(self.idx_test, self.idx_test)]

        self.labels_train = self.labels_full[self.idx_train]
        self.labels_val = self.labels_full[self.idx_val]
        self.labels_test = self.labels_full[self.idx_test]

        self.feat_train = self.feat_full[self.idx_train]
        self.feat_val = self.feat_full[self.idx_val]
        self.feat_test = self.feat_full[self.idx_test]

    def to(self, device):
        """Move data to the specified device."""
        self.feat_full = self.feat_full.to(device)
        self.labels_full = self.labels_full.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)

        self.feat_train = self.feat_train.to(device)
        self.feat_val = self.feat_val.to(device)
        self.feat_test = self.feat_test.to(device)

        return self

    def pyg_saint(self, data):
        # reference type
        # pyg format use x,y,edge_index
        if hasattr(data, 'x'):
            self.x = data.x
            self.y = data.y
            self.feat_full = data.x
            self.labels_full = data.y
            self.adj_full = ei2csr(data.edge_index, data.x.shape[0])
            self.edge_index = data.edge_index
            self.sparse_adj = SparseTensor.from_edge_index(data.edge_index)
        # saint format use feat,labels,adj
        elif hasattr(data, 'feat_full'):
            self.adj_full = data.adj_full
            self.feat_full = data.feat_full
            self.labels_full = data.labels_full
            self.edge_index = csr2ei(data.adj_full)
            self.sparse_adj = SparseTensor.from_edge_index(self.edge_index)
            self.x = data.feat_full
            self.y = data.labels_full
        return data

    def retrieve_class(self, c, num=256):
        # change the initialization strategy here
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, args, num=256):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if args.setting == 'trans':
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        elif args.nlayers == 2:
            sizes = [15, 8] if args.dataset in ['reddit', 'flickr'] else [10, 5]
        elif args.nlayers == 3:
            sizes = [15, 10, 5]
        elif args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        elif args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    self.samplers.append([])
                    continue
                self.samplers.append(
                    NeighborSampler(
                        adj,
                        node_idx=node_idx,
                        sizes=sizes,
                        batch_size=num,
                        num_workers=8,
                        return_e_id=False,
                        num_nodes=adj.size(0),
                        shuffle=True,
                    )
                )
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch.astype(np.int64))

        return out


    def reset(self):
        self.samplers = None
        self.class_dict2 = None
        self.labels_syn, self.feat_syn, self.adj_syn = None, None, None

class NodeClassificationDataset(object):
    def __init__(self, root: str, feat_name: str, verbose: bool=True, device: str="cpu"):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.device = device
        
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Feature name: {self.feat_name}')
            print(f'Device: {self.device}')
        
        # Load edges
        edge_path = os.path.join(root, 'nc_edges-nodeid.pt')
        
        self.edge = torch.tensor(torch.load(edge_path), dtype=torch.int64).to(self.device)
        
        # Load features
        feat_path = os.path.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location=self.device)
        self.num_nodes = feat.shape[0]
        print("feat shape:", feat.shape)
        
        # Create edge_index in PyG format (2 x num_edges)
        edge_index = self.edge.t().contiguous()
        
        # Add reverse edges to make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
        
        # Load labels
        labels_path = os.path.join(root, 'labels-w-missing.pt')
        self.labels = torch.tensor(torch.load(labels_path), dtype=torch.int64).to(self.device)
        
        # Load node splits
        node_split_path = os.path.join(root, 'split.pt')
        self.node_split = torch.load(node_split_path)
        
        # Create masks
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)

        train_mask[self.node_split['train_idx']] = True
        val_mask[self.node_split['val_idx']] = True
        test_mask[self.node_split['test_idx']] = True
        
        if self.name in ['books_nc', 'books_lp'] :
            self.num_classes = len(torch.unique(self.labels))-1
        else:
            self.num_classes = 12
        
        # Create PyG Data object
        self.data = Data(
            x=feat,
            edge_index=edge_index,
            y=self.labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=self.num_nodes
        )
        
    def get_idx_split(self):
        return self.node_split
    
    @property 
    def graph(self):
        """Return the PyG Data object for compatibility"""
        return self.data
    
    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.data
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
