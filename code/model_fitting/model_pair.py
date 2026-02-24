import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch.nn import Embedding, Linear, LSTM
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from .layers import Protein_PNAConv, DrugProteinConv, PosLinear, GCNCluster
from .pool import GraphMultisetTransformer

## for cluster
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph, softmax
from .protein_pool0 import dense_mincut_pool,MotifPool
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv, APPNP, SGConv

EPS = 1e-15
class net(torch.nn.Module):
    def __init__(self, prot_deg,
                 # MOLECULE
                 mol_in_channels=43, prot_in_channels=40, prot_evo_channels=1280,
                 hidden_channels=200, pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,
                 K=[3, 5, 10],
                 t=1,
                 # training
                 heads=5,
                 dropout=0,
                 dropout_attn_score=0.2,
                 drop_atom=0,
                 drop_residue=0,
                 dropout_cluster_edge=0,
                 gaussian_noise=0,
                 device='cuda:0'):
        super(net, self).__init__()

        self.hidden_channels = hidden_channels
        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, hidden_channels)

        ### MOLECULE and PROTEIN
        self.prot_convs = torch.nn.ModuleList()
        self.atom_update = torch.nn.ModuleList()
        self.inter_convs = torch.nn.ModuleList()

        self.num_cluster = K
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.mol_norms = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()
        self.res_lin = torch.nn.ModuleList()
        self.mol_lin = torch.nn.ModuleList()
        self.atom_embed_total2 = torch.nn.ModuleList()
        self.atom_embed_total1 = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.prot_convs.append(Protein_PNAConv(prot_deg, hidden_channels, edge_channels=hidden_channels, pre_layers=pre_layers, post_layers=post_layers,
                                                   aggregators=aggregators, scalers=scalers, num_towers=heads, dropout=dropout))
            self.atom_update.append(Linear(hidden_channels, hidden_channels))

            self.cluster.append(GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]]))
            self.inter_convs.append(DrugProteinConv(atom_channels=hidden_channels, residue_channels=hidden_channels, heads=heads, t=t,dropout_attn_score=dropout_attn_score))

            #self.mol_pools.append(MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom))

            self.res_lin.append(Linear(hidden_channels, hidden_channels))
            self.mol_lin.append(Linear(hidden_channels, hidden_channels))
            self.atom_embed_total2.append(Linear(hidden_channels*2, hidden_channels))
            self.atom_embed_total1.append(Linear(hidden_channels, hidden_channels))


        self.dropout = dropout
        self.drop_atom = drop_atom
        self.device = device

        #self.seq_embed_aa = torch.nn.Linear(prot_in_channels, hidden_channels)
        self.seq_embed_esm = torch.nn.Linear(prot_in_channels, hidden_channels*2)
        self.seq_embed_evo = torch.nn.Linear(prot_evo_channels, hidden_channels*2)
        self.seq_embed_evo1 = torch.nn.Linear(hidden_channels*4, hidden_channels)
        self.seq_embed_evo2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        #pan1105
        self.atom_embed= torch.nn.Linear(767, hidden_channels)
        self.atom_embed2 = torch.nn.Linear(hidden_channels, hidden_channels)


        self.norm = torch.nn.LayerNorm(hidden_channels) #
        self.GN = GraphNorm(hidden_channels)
        #self.enz_cluster=GCNCluster([hidden_channels, hidden_channels, 10], in_norm=True) #self.num_cluster[idx]]

        self.atom_feat_embed = Linear(mol_in_channels, hidden_channels)
        self.atom_type_embed = Embedding(20, hidden_channels)
        self.atom_type_embed2 = Linear(hidden_channels // 2, hidden_channels // 2)
        self.atom_feat_embed2 = Linear(hidden_channels//2, hidden_channels)

        #self.mol_pool= MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom)
        #self.enz_mol_inter = DrugProteinConv(atom_channels=hidden_channels, residue_channels=hidden_channels, heads=heads, t=t,dropout_attn_score=dropout_attn_score)

        #self.res_lin = Linear(hidden_channels, hidden_channels, bias=False)
        self.res_lin2 = Linear(hidden_channels, hidden_channels, bias=False)
        self.res_attn_lin = PosLinear(heads, 1, bias=False,init_value=1 / heads)  # (heads * total_layer)) PositiveLinear(heads, 1, bias=False)#
        self.res_attn_lin1 = PosLinear(heads, 1, bias=False,init_value=1 / heads) #PositiveLinear(heads, 1, bias=False)#

        self.classifier0=nn.Linear(hidden_channels*4, 512) #1024
        #self.classifier0=nn.Linear(hidden_channels*(3*total_layer+1), 512) #1024
        self.classifier1 = nn.Linear(512, 128)
        self.classifier2 = nn.Linear(128, 1)

        self.enz_fea_lin = Linear(hidden_channels * 2 * total_layer, hidden_channels)
        self.mol_fea_lin = Linear(hidden_channels * total_layer, hidden_channels)
        self.mol_fea_lin1 = Linear(hidden_channels , hidden_channels) #* total_layer
        self.mol_fea_lin2 = Linear(hidden_channels * total_layer, hidden_channels)
        self.mol_fea_lin3 = Linear(hidden_channels * total_layer, hidden_channels)
        #self.smi_attention_poc = EncoderLayer(hidden_channels, hidden_channels, 0.1, 0.1, 4)  # 注意力机制
        self.pool = GraphMultisetTransformer(hidden_channels, 256, hidden_channels, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'], num_heads=8,layer_norm=True)

        #0424pan
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

    def forward(self, mol_x, mol_x_feat, total_fea, bond_x, atom_edge_index,
                # Protein
                residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
                # Mol-Protein Interaction batch
                mol_batch=None, prot_batch=None, clique_batch=None):

        spectral_loss = torch.tensor(0.).to(self.device)
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)
        residue_scores = []

        # Init variables - PROTEIN Featurize
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)
        residue_ini_0 = F.relu(self.seq_embed_evo(residue_evo_x))
        residue_ini_1 = F.relu(self.seq_embed_esm(residue_x))
        residue_ini = torch.cat([residue_ini_0, residue_ini_1], dim=-1)
        residue_x = F.relu(self.seq_embed_evo1(residue_ini))

        # init variables - MOLECULE Featurize
        atom_x = self.atom_type_embed(mol_x.squeeze())+ F.relu(self.atom_feat_embed(mol_x_feat))#self.norm(self.atom_feat_embed2()  # (519,200)
        total_fea = F.relu(self.atom_embed(total_fea))
        total_fea = self.norm(self.atom_embed2(total_fea))

        enz_feas1 = []
        enz_feas2 = []
        mol_feas = []
        cluster_feas = []

        for idx in range(self.total_layer):
            residue_x = self.GN(residue_x, prot_batch)
            atom_x = self.GN(atom_x, mol_batch)

            # cluster residues
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            residue_enz = global_max_pool(residue_x, prot_batch) #(32,200)
            enz_feas1.append(residue_enz) #现在residue_enz 大一个量级

            s = self.cluster[idx](residue_x, residue_edge_index)  # (12329,cluster_num) GCN聚类
            #print("CLUSTER IS",s[:50])
            s, _ = to_dense_batch(s, prot_batch)  # (32,903,cluster_num)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)  ##(32,903,200)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)  # (32,903,903)
            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, s, residue_mask, None)  # S(32,903,cluster_num) cluster_x(32,5,200) residue_adj(B,5,5)
            cluster_x = self.norm(cluster_x)  # cluster_x(32,3,200)
            ortho_loss += o_loss
            cluster_loss += cl_loss

            # Pool Drug ball
            atom_x = F.relu(self.atom_update[idx](atom_x))#F.leaky_relu()
            mol_x= global_add_pool(atom_x, mol_batch) #self.mol_pools[idx](atom_x, mol_batch) #(32,200)
            mol_x = self.norm(mol_x)  # (B,200) #norm之后会变大一个量级

            mol_x = torch.cat([total_fea, mol_x], dim=-1)
            mol_x = F.relu(self.atom_embed_total2[idx](mol_x))
            mol_x = self.atom_embed_total1[idx](mol_x)  # )
            mol_x = self.norm(mol_x)

            # MOLECULE-PROTEIN Layers
            # connect drug and protein cluster
            batch_size = s.size(0)
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)  # cluster_x(160,200)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device) #(0001112)
            p2m_edge_index = torch.stack([torch.arange(batch_size * self.num_cluster[idx]),torch.arange(batch_size).repeat_interleave(self.num_cluster[idx])]).to(self.device)

            ## model interative relationship
            mol_x, cluster_x, inter_attn = self.inter_convs[idx](mol_x, cluster_x, p2m_edge_index)  # mol_x(32,200), cluster_x(192,200), inter_attn((2,B*5),(B*5,5)) #mol_x太小而cluster太大

            #enz_cluster = global_max_pool(cluster_x,cluster_residue_batch)
            inter_attn = inter_attn[1] #(B*cluster_num,heads_num) #pan0715 inter_attn已经不对了
            #print("33333",inter_attn)
            mol_feas.append(mol_x)

            atom_x = atom_x + F.relu(self.mol_lin[idx](mol_x)[mol_batch])#self.norm()  # drug -> atom (519,200)

            #pan0715
            cluster_score = softmax(self.res_attn_lin1(inter_attn), cluster_residue_batch)
            pool_cluster = global_add_pool(cluster_x * cluster_score, cluster_residue_batch)
            cluster_feas.append(pool_cluster)

            cluster_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)  # (32,3,200)
            inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)  # (B,3,head5)

            residue_x = residue_x + F.relu((self.res_lin[idx]((s @ cluster_hx)[residue_mask]))) #self.norm() # cluster -> residue (429,200)

            # pan0715 应为1 现在0.02 --> pool_enz太小了,两到三个数量级
            residue_score = self.res_attn_lin( (s @ inter_attn)[residue_mask])
            #print(residue_score.size())
            residue_score = softmax(residue_score, prot_batch) #
            residue_scores.append(residue_score)

            pool_enz = global_add_pool(residue_x * residue_score, prot_batch)
            enz_feas2.append(pool_enz)

        g_level_feat = self.pool(residue_x, prot_batch,  residue_edge_index)

        mol_feas = torch.cat(mol_feas, dim=-1)
        #enz_feas1 = torch.cat(enz_feas1, dim=-1)
        enz_feas2 = torch.cat(enz_feas2, dim=-1)
        clu_fea = torch.cat(cluster_feas,dim=-1)

        mol_x = F.relu(self.mol_fea_lin(mol_feas))#F.leaky_relu()
        enz_feas1 = F.relu(self.mol_fea_lin1(g_level_feat))
        enz_feas2 = F.relu(self.mol_fea_lin2(enz_feas2))  # F.leaky_relu()
        clu_fea = F.relu(self.mol_fea_lin3(clu_fea))  # F.leaky_relu()
        mol_prot_feat = torch.cat([enz_feas1, enz_feas2,clu_fea,mol_x], dim=-1)  #

        reg_pred = F.relu((self.classifier0(mol_prot_feat)))
        reg_pred = F.relu((self.classifier1(reg_pred)))
        reg_pred =  self.classifier2(reg_pred)
        #binary_pred = nn.Sigmoid()(reg_pred)


        attention_dict = {
            'residue_final_score': residue_scores,
            'residue_layer_scores': residue_scores,
            'drug_atom_index': mol_batch,
            'protein_residue_index': prot_batch,
            'interaction_fingerprint': mol_prot_feat}

        return reg_pred, spectral_loss, ortho_loss, cluster_loss, attention_dict

    def temperature_clamp(self):
        pass
        # with torch.no_grad():
        #     for m in self.cluster:
        #         m.logit_scale.clamp_(0, math.log(100))

    def connect_mol_prot(self, mol_batch, prot_batch):
        mol_num_nodes = mol_batch.size(0)
        prot_num_nodes = prot_batch.size(0)
        mol_adj = mol_batch.reshape(-1, 1).repeat(1, prot_num_nodes)
        pro_adj = prot_batch.repeat(mol_num_nodes, 1)

        m2p_edge_index = (mol_adj == pro_adj).nonzero(as_tuple=False).t().contiguous()

        return m2p_edge_index

    def freeze_backbone_optimizers(self, finetune_module, weight_decay, learning_rate, betas, eps,amsgrad):  ## only for fineTune Pretrain model
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...

                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################
                if not any([mn.startswith(name) for name in finetune_module]):
                    p.requires_grad = False
                    continue
                else:
                    p.requires_grad = True
                    print(fpn, ' will be finetuned')
                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################

                if pn.endswith('bias') or pn.endswith('mean_scale'):  # or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,torch.nn.LSTM, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias') or pn.endswith('mean_scale'):  # or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn[pn.find('.')+1:].startswith('weight')and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn[pn.find('.')+1:].startswith('bias'):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!"  % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer


def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    Returns an 径向基函数 Radial Basis Function - RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device))
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(edge_index, batch):
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def compute_connectivity(edge_index, batch):  ## for numerical stability (i.e. we cap inv_con at 100)

    edges_by_batch = unbatch_edge_index(edge_index, batch)

    nodes_counts = torch.unique(batch, return_counts=True)[1]

    connectivity = torch.tensor([nodes_in_largest_graph(e, n) for e, n in zip(edges_by_batch, nodes_counts)])
    isolation = torch.tensor([isolated_nodes(e, n) for e, n in zip(edges_by_batch, nodes_counts)])

    return connectivity, isolation


def nodes_in_largest_graph(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(adj)

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return subset.sum() / num_nodes


def isolated_nodes(edge_index, num_nodes):
    r"""Find isolate nodes """
    edge_attr = None

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 0

    return mask.sum() / num_nodes


def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    ## ensure no graph is totally dropped out
    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []

    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)

    node_mask = torch.cat(node_mask_list)

    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask


class LinkAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.input_dim = input_dim
        self.query = torch.nn.Linear(input_dim, n_heads)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x): #（197，200）
        query = self.query(x).transpose(0, 1) #（5，197）
        a = self.softmax(query)
        out = torch.matmul(a, x) #（5，200）
        out = torch.sum(out, dim=0).reshape(-1,self.input_dim)
        return out, a