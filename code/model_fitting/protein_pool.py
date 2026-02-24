import torch

EPS = 1e-15
from torch_scatter import scatter
from .layers0610 import *


def dense_mincut_pool(x, adj, s, mask=None, cluster_drop_node=None):
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
    loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)

        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    # out_loss = mincut_loss + ortho_loss

    return s, out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.dense.mincut_pool import _rank3_trace

EPS = 1e-15


def dense_dmon_pool(x, adj, s, mask=None):
    r"""
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in
            \mathbb{R}^{B \times N \times F}` with batch-size
            :math:`B`, (maximum) number of nodes :math:`N` for each graph,
            and feature dimension :math:`F`.
            Note that the cluster assignment matrix
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
            being created within this method.
        adj (Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
    """

    # x = x.unsqueeze(0) if x.dim() == 2 else x
    # adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    # s = s.unsqueeze(0) if s.dim() == 2 else s

    # (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    # s_out = torch.softmax(s, dim=-1)

    # if mask is not None:
    #     mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    #     x, s = x * mask, s_out * mask

    # out = torch.matmul(s.transpose(1, 2), x)
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    s = torch.softmax(s, dim=-1)
    s_out = s

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Spectral loss:
    degrees = torch.einsum('ijk->ik', adj).transpose(0, 1)
    m = torch.einsum('ij->', degrees)

    ca = torch.matmul(s.transpose(1, 2), degrees)
    cb = torch.matmul(degrees.transpose(0, 1), s)

    normalizer = torch.matmul(ca, cb) / 2 / m
    decompose = out_adj - normalizer
    spectral_loss = -_rank3_trace(decompose) / 2 / m
    spectral_loss = torch.mean(spectral_loss)

    # Orthogonality regularization:
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Cluster loss:
    cluster_loss = torch.norm(torch.einsum(
        'ijk->ij', ss)) / adj.size(1) * torch.norm(i_s) - 1

    # Fix and normalize coarsened adjacency matrix:
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s_out, out, out_adj, spectral_loss, ortho_loss, cluster_loss


import torch

EPS = 1e-15


def simplify_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator from the `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}
        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and the following auxiliary objective:
    .. math::
        \mathcal{L} = - {\mathrm{Tr}(\sqrt{\mathbf{S}^{\top} \mathbf{S}})}
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)
    s_out = s

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s_out, out, out_adj, loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

class MotifPool(torch.nn.Module):
    def __init__(self, hidden_dim, heads, dropout_attn_score=0, dropout_node_proba=0):
        super().__init__()
        assert hidden_dim % heads == 0

        self.lin_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        hidden_dim = hidden_dim // heads

        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads):
            self.score_proj.append(MLP([hidden_dim, hidden_dim * 2, 1]))

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout_node_proba = dropout_node_proba
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(self, x, x_clique):
        H = self.heads
        C = self.hidden_dim
        ## residual connection + atom2clique
        hx_clique = scatter(x[row], col, dim=0, dim_size=x_clique.size(0), reduce='mean')
        x_clique = x_clique + F.relu(self.lin_proj(hx_clique))
        ## GNN scoring
        score_clique = x_clique.view(-1, H, C)
        score = torch.cat([mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj)], dim=-1)
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)

        ## multihead aggregation of drug feature
        scaling_factor = 1.
        _, _, clique_drop_mask = dropout_node(clique_edge_index, self.dropout_node_proba, x_clique.size(0),
                                              clique_batch, self.training)
        scaling_factor = 1. / (1. - self.dropout_node_proba)

        drug_feat = x_clique.view(-1, H, C) * alpha.view(-1, H, 1)
        drug_feat = drug_feat.view(-1, H * C) * clique_drop_mask.view(-1, 1)
        drug_feat = global_add_pool(drug_feat, clique_batch) * scaling_factor

        return drug_feat, x_clique, alpha

    """def forward(self, x, x_clique, atom2clique_index, clique_batch, clique_edge_index):
        row, col = atom2clique_index
        H = self.heads
        C = self.hidden_dim
        ## residual connection + atom2clique
        hx_clique = scatter(x[row], col, dim=0, dim_size=x_clique.size(0), reduce='mean')
        x_clique = x_clique + F.relu(self.lin_proj(hx_clique))
        ## GNN scoring
        score_clique = x_clique.view(-1, H, C)
        score = torch.cat([mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj)], dim=-1)
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)

        ## multihead aggregation of drug feature
        scaling_factor = 1.
        _, _, clique_drop_mask = dropout_node(clique_edge_index, self.dropout_node_proba, x_clique.size(0), clique_batch, self.training)
        scaling_factor = 1. / (1. - self.dropout_node_proba)

        drug_feat = x_clique.view(-1, H, C) * alpha.view(-1, H, 1)
        drug_feat = drug_feat.view(-1, H * C) * clique_drop_mask.view(-1, 1)
        drug_feat = global_add_pool(drug_feat, clique_batch) * scaling_factor

        return drug_feat, x_clique, alpha"""