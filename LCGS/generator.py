import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Generator(nn.Module):
    def __init__(self, config, adj):
        super(Generator, self).__init__()
        self.dropout = config.dropout
        self.self_loop = config.self_loop
        self.diag_coeff = config.diag_coeff
        self.epsilon = config.epsilon

        self.adj_para = nn.Parameter(torch.zeros_like(adj))
        self.lower_tri_mask = torch.tril(torch.ones(adj.shape), diagonal=-1).to(adj)
        if self.self_loop:
            self.upper_tri_mask = self.lower_tri_mask.transpose(1, 0) + torch.eye(int(adj.shape[0])).to(adj)
        else:
            self.upper_tri_mask = self.lower_tri_mask.transpose(1, 0)

        self.adj_para.data = self.get_symmetric_normalize(adj) * self.upper_tri_mask

    def get_symmetric_normalize(self, adj, diag=1.):
        tilde_A = adj + diag * torch.eye(int(adj.shape[0])).to(adj)
        tilde_D = (tilde_A.sum(1) + self.epsilon)
        # if sum(tilde_D == 0):
        #     print("tilde_D.has_zero:", sum(tilde_D == 0))
        sqrt_tildD = 1. / torch.sqrt(tilde_D)

        hat_A = tilde_A * sqrt_tildD.view(-1, 1) * sqrt_tildD
        return hat_A

    def get_row_normalize(self, adj, diag=1.):
        tilde_A = adj + diag * torch.eye(int(adj.shape[0])).to(adj)
        tilde_D = (tilde_A.sum(1) + self.epsilon)

        inv_tildD = 1. / tilde_D
        hat_A = tilde_A * inv_tildD.view(-1, 1)
        return hat_A

    def upper_constraint(self):
        return self.adj_para.data.clamp_(min=0., max=1.).mul_(self.upper_tri_mask)

    def constraint(self):
        return self.adj_para.data.clamp_(min=0., max=1.)

    def forward(self, mask=False):
        """

        :param x: feature vector, (n, f_dim)
        :param adj: init adjacency matrix
        :return: adj (normalized adj)
        """

        self.upper_constraint()
        if mask:
            mask_adj = F.dropout(self.adj_para, p=self.dropout)
            symnorm_adj = mask_adj + mask_adj.transpose(1, 0) * self.lower_tri_mask
        else:
            symnorm_adj = self.adj_para + self.adj_para.transpose(1, 0) * self.lower_tri_mask
        n_adj = self.get_symmetric_normalize(symnorm_adj, diag=self.diag_coeff)
        return n_adj, True