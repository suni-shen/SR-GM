from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *
import torch.nn.functional as F
import time


class GCDM(GCondBase):
    def __init__(self, setting, data, args, **kwargs):
        super(GCDM, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        pge = self.pge
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        model = eval(args.condense_model)(self.d, args.hidden,
                                          data.nclass, args).to(self.device)

        feat_syn = self.feat_syn

        best_val = 0
        bar = trange(args.epochs)
        for it in bar:
            model.initialize()
            model_parameters = list(model.parameters())
            self.optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()
            with torch.no_grad():
                emb_real,_ = model.forward(features,adj, output_layer_features=True)

            loss_avg = 0
            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                adj_syn = self.adj_syn
                emb_syn, _ = model.forward(feat_syn,adj_syn, output_layer_features=True)
                loss_emb = 0
                for i in range(len(emb_syn)):
                    if i == args.nlayers-1:
                        break
                    for c in range(data.nclass):
                        if c not in self.num_class_dict:
                            continue
                        coeff = self.num_class_dict[c] / self.nnodes_syn

                        st_id, ed_id = self.syn_class_indices[c]
                        num_syn_samples = emb_syn[i][st_id:ed_id].shape[0]
                        class_mask = (data.labels_train == c)
                        real_indices = class_mask.nonzero(as_tuple=False).squeeze()

                        num_real_samples = real_indices.shape[0]

                        selected_indices = real_indices[torch.randperm(num_real_samples)[:num_syn_samples]]

                        if args.setting == 'trans':
                            emb_real_selected = emb_real[i][data.idx_train][selected_indices]
                            emb_real_class = emb_real[i][data.idx_train][class_mask]
                        else:
                            emb_real_selected = emb_real[i][selected_indices]
                            emb_real_class = emb_real[i][class_mask]
                        
                        emb_syn_selected = emb_syn[i][st_id:ed_id]

                        loss_emb += coeff * dist(emb_real_selected, emb_syn_selected, method=args.dis_metric)
                loss_avg += loss_emb.item()
                
                if args.debug:
                    print("loss_avg: ", loss_avg)
                
                # --- SR-GM ---
                if args.sr:
                    output_syn_sr = model.forward(feat_syn, adj_syn)
                    loss_syn_sr = F.nll_loss(output_syn_sr,labels_syn)
                    g_feat_syn_raw = torch.autograd.grad(loss_syn_sr, feat_syn, retain_graph=True)[0]
                    
                    L_syn = self.get_laplacian(adj_syn)
                    
                    # --- Component I: Gradient Decoupling ---
                    g_text = g_feat_syn_raw[:, :args.d_text]
                    g_image = g_feat_syn_raw[:, args.d_text:]
                    
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
                    loss_sr = torch.trace(decoupled_grads.T @ L_syn @ decoupled_grads)
                    
                    if args.debug:
                        print("lambad*loss_sr is {}".format(args.lambad*loss_sr.item()))
                        
                    loss_emb = loss_emb + args.lambad * loss_sr

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss_emb.backward()

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner

                for _ in range(inner_loop):
                    emb_syn, _ = model.forward(feat_syn_inner_norm, adj_syn_inner_norm, output_layer_features=True)
                    loss_inner = 0
                    for i in range(len(emb_syn)):
                        if i == args.nlayers-1:
                            break
                        for c in range(data.nclass):
                            if c not in self.num_class_dict:
                                continue
                            coeff = self.num_class_dict[c] / self.nnodes_syn

                            st_id, ed_id = self.syn_class_indices[c]
                            num_syn_samples = emb_syn[i][st_id:ed_id].shape[0]
                            class_mask = (data.labels_train == c)
                            real_indices = class_mask.nonzero(as_tuple=False).squeeze()

                            num_real_samples = real_indices.shape[0]

                            selected_indices = real_indices[torch.randperm(num_real_samples)[:num_syn_samples]]

                            if args.setting == 'trans':
                                emb_real_selected = emb_real[i][data.idx_train][selected_indices]
                                emb_real_class = emb_real[i][data.idx_train][class_mask]
                            else:
                                emb_real_selected = emb_real[i][selected_indices]
                                emb_real_class = emb_real[i][class_mask]


                            emb_syn_selected = emb_syn[i][st_id:ed_id]

                            loss_inner += coeff * dist(emb_real_selected, emb_syn_selected, method=args.dis_metric)
                    self.optimizer_model.zero_grad()
                    loss_inner.backward()
                    self.optimizer_model.step()
                with torch.no_grad():
                    emb_real,_ = model.forward(features,adj, output_layer_features=True)
                    
            loss_avg /= outer_loop
            bar.set_postfix({'loss': loss_avg})

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
        
def dist(x, y, method='l1'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
    return dist_

