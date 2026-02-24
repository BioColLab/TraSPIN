import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch.nn import Embedding, Linear, LSTM
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from .layers import Protein_PNAConv, GCNCluster
from .protein_pool import dense_mincut_pool
## for cluster
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph

from torch_geometric.nn.norm import GraphNorm


EPS = 1e-15
class net(torch.nn.Module):
    def __init__(self, num_class,prot_deg,
                 prot_in_channels=1280, prot_evo_channels=1024,
                 hidden_channels=200, pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 total_layer=3,
                 K=[3, 10, 30],
                 # training
                 heads=5,
                 dropout=0,
                 device='cuda:0'):
        super(net, self).__init__()

        self.num_cluster = K
        self.cluster = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()
        self.res_lin = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):

            self.prot_convs.append(Protein_PNAConv(prot_deg, hidden_channels, edge_channels=hidden_channels, pre_layers=pre_layers, post_layers=post_layers,
                                                   aggregators=aggregators, scalers=scalers, num_towers=heads, dropout=dropout))

            self.cluster.append(GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]]))
            self.res_lin.append(Linear(hidden_channels, hidden_channels))


        self.dropout = dropout
        self.device = device

        #self.seq_embed_aa = torch.nn.Linear(prot_in_channels, hidden_channels)
        self.seq_embed_esm = torch.nn.Linear(prot_in_channels, hidden_channels*2)
        self.seq_embed_evo = torch.nn.Linear(prot_evo_channels, hidden_channels*2)
        self.seq_embed_evo1 = torch.nn.Linear(hidden_channels*4, hidden_channels)
        self.seq_embed_evo2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)

        self.norm = torch.nn.LayerNorm(hidden_channels) #
        self.GN = GraphNorm(hidden_channels)

        self.mol_fea_lin1 = Linear(hidden_channels * total_layer, hidden_channels)
        self.mol_fea_lin2 = Linear(hidden_channels * total_layer, hidden_channels)
        self.mol_fea_lin3 = Linear(hidden_channels * total_layer, hidden_channels)

        self.classifier0=nn.Linear(hidden_channels*3, 512) #1024
        self.classifier1 = nn.Linear(512, 128)
        self.classifier2 = nn.Linear(128, num_class)

        #0424pan
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)


    def forward(self, residue_x, residue_evo_x, residue_edge_index, residue_edge_weight, prot_batch=None):

        spectral_loss = torch.tensor(0.).to(self.device)
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)

        # Init variables - PROTEIN Featurize
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)
        residue_ini_0 = F.relu(self.seq_embed_evo(residue_evo_x))
        residue_ini_1 = F.relu(self.seq_embed_esm(residue_x))
        residue_ini = torch.cat([residue_ini_0, residue_ini_1], dim=-1)
        residue_x = F.relu(self.seq_embed_evo1(residue_ini))

        enz_feas1 = []
        enz_feas2 = []
        cluster_feas = []

        for idx in range(self.total_layer):
            residue_x = self.GN(residue_x, prot_batch)

            # cluster residues
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            residue_enz = global_max_pool(residue_x, prot_batch) #(32,200)
            enz_feas1.append(residue_enz) #现在residue_enz 大一个量级

            s = self.cluster[idx](residue_x, residue_edge_index)  # (12329,cluster_num) GCN聚类
            s, _ = to_dense_batch(s, prot_batch)  # (32,903,cluster_num)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)  ##(32,903,200)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)  # (32,903,903)
            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, s, residue_mask, None)  # S(32,903,cluster_num) cluster_x(32,5,200) residue_adj(B,5,5)
            cluster_x = self.norm(cluster_x)  # cluster_x(32,3,200)
            ortho_loss += o_loss
            cluster_loss += cl_loss

            # MOLECULE-PROTEIN Layers
            batch_size = s.size(0)
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)  # cluster_x(160,200)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device) #(0001112)

            #pan0715 应为1 现在0.3 --> pool_cluster太小了
            pool_cluster = self.norm(global_max_pool(cluster_x, cluster_residue_batch))
            cluster_feas.append(pool_cluster)
            cluster_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)  # (32,3,200)

            residue_x = residue_x + F.relu((self.res_lin[idx]((s @ cluster_hx)[residue_mask]))) #self.norm() # cluster -> residue (429,200)
            # pan0715 应为1 现在0.02 --> pool_enz太小了,两到三个数量级
            pool_enz = self.norm(global_max_pool(residue_x, prot_batch))
            enz_feas2.append(pool_enz)

        enz_feas1 = torch.cat(enz_feas1, dim=-1)
        enz_feas2 = torch.cat(enz_feas2, dim=-1)
        clu_fea = torch.cat(cluster_feas,dim=-1)

        enz_feas1 = F.relu(self.mol_fea_lin1(enz_feas1))
        enz_feas2 = F.relu(self.mol_fea_lin2(enz_feas2))  # F.leaky_relu()
        clu_fea = F.relu(self.mol_fea_lin3(clu_fea))  # F.leaky_relu()
        transport_feat = torch.cat([enz_feas1, enz_feas2, clu_fea], dim=-1)  #clu_fea

        reg_pred = F.relu((self.classifier0(transport_feat)))
        reg_pred = F.relu((self.classifier1(reg_pred)))
        reg_pred = self.classifier2(reg_pred)

        return reg_pred, spectral_loss, ortho_loss, cluster_loss

    def temperature_clamp(self):
        pass

    def get_embedding(self, residue_x, residue_evo_x, residue_edge_index,
                      residue_edge_weight, prot_batch=None):
        """提取 classifier1 输出的 128 维 embedding"""
        spectral_loss = torch.tensor(0.).to(self.device)
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)

        # === 完全复制 forward 的特征提取部分 ===
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0,
                                 D_count=self.prot_edge_dim, device=self.device)
        residue_ini_0 = F.relu(self.seq_embed_evo(residue_evo_x))
        residue_ini_1 = F.relu(self.seq_embed_esm(residue_x))
        residue_ini = torch.cat([residue_ini_0, residue_ini_1], dim=-1)
        residue_x = F.relu(self.seq_embed_evo1(residue_ini))

        enz_feas1, enz_feas2, cluster_feas = [], [], []

        for idx in range(self.total_layer):
            residue_x = self.GN(residue_x, prot_batch)
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            residue_enz = global_max_pool(residue_x, prot_batch)
            enz_feas1.append(residue_enz)

            s = self.cluster[idx](residue_x, residue_edge_index)
            s, _ = to_dense_batch(s, prot_batch)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(
                residue_hx, residue_adj, s, residue_mask, None)
            cluster_x = self.norm(cluster_x)
            batch_size = s.size(0)
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(
                self.num_cluster[idx]).to(self.device)
            pool_cluster = self.norm(global_max_pool(cluster_x, cluster_residue_batch))
            cluster_feas.append(pool_cluster)
            cluster_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)
            residue_x = residue_x + F.relu(self.res_lin[idx]((s @ cluster_hx)[residue_mask]))
            pool_enz = self.norm(global_max_pool(residue_x, prot_batch))
            enz_feas2.append(pool_enz)

        enz_feas1 = F.relu(self.mol_fea_lin1(torch.cat(enz_feas1, dim=-1)))
        enz_feas2 = F.relu(self.mol_fea_lin2(torch.cat(enz_feas2, dim=-1)))
        clu_fea = F.relu(self.mol_fea_lin3(torch.cat(cluster_feas, dim=-1)))
        transport_feat = torch.cat([enz_feas1, enz_feas2, clu_fea], dim=-1)

        emb = F.relu(self.classifier0(transport_feat))
        emb = F.relu(self.classifier1(emb))  # ← 128 维，这就是 embedding
        return emb


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
