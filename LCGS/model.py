import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dense_gcn_2l_tensor(x, adj, y, hidden_dim=16, num_layer=2, dropout=None, name='GCN'):
    """builds up a simple GCN model with dense adjacency matrix"""
    f_dim = int(x.shape[1])
    out_dim = int(y.shape[1])

    assert len(x.shape) == 2
    assert len(y.shape) == 2
    tilde_A = adj + torch.eye(int(adj.shape[0]))
    tilde_D = tilde_A.sum(0)
    sqrt_tildD = 1. / torch.sqrt(tilde_D)
    daBda = lambda _b, _a: torch.mm(_b, _a.transpose(1,0)).mm(_b)
    hatA = daBda(sqrt_tildD, tilde_A)

    W = []
    for i in range(num_layer):
        if i == 0:
            W.append(torch.nn.Parameter(torch.randn(f_dim, hidden_dim)))
        elif i == num_layer - 1:
            W.append(torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim)))
        else:
            W.append(torch.nn.Parameter(torch.randn(hidden_dim, out_dim)))

    representation = F.relu(hatA @ x @ W[0])
    if dropout is not None:
        representation = F.dropout(representation, dropout)

    for i in range(1, num_layer-1):
        representation = F.relu(hatA @ representation @ W[i])
        if dropout is not None:
            representation = F.dropout(representation, dropout)
    out = hatA @ representation @ W[-1]

    return out, W, representation

class GCN(nn.Module):
    def __init__(self, shape, out_dim, hidden_dim=16, num_layer=2, dropout=0, device=None):
        super(GCN, self).__init__()
        assert device is not None
        self.shape = shape
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.dropout = dropout
        self.f_dim = shape[1]
        self.out_dim = out_dim

        self.W = []
        for i in range(self.num_layer):
            if i == 0:
                self.W.append(torch.nn.Parameter(torch.randn(self.f_dim, self.hidden_dim, device=self.device)))
            elif i == num_layer - 1:
                self.W.append(torch.nn.Parameter(torch.randn(self.hidden_dim, self.out_dim, device=self.device)))
            else:
                self.W.append(torch.nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, device=self.device)))

    def glorot(self):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (self.shape[0] + self.shape[1]))
        for i in range(self.num_layer):
            nn.init.uniform_(self.W[i], a=-init_range, b=init_range)

    def xavier_uniform(self):
        for i in range(self.num_layer):
            nn.init.xavier_uniform_(self.W[i])

    def init_weight(self):
        self.xavier_uniform()

    def forward(self, x, adj, W=None, normalized=False):
        assert len(x.shape) == 2
        if normalized is False:
            tilde_A = adj + torch.eye(int(adj.shape[0])).to(self.device)
            tilde_D = tilde_A.sum(1)
            if sum(tilde_D == 0):
                print("tilde_D.has_zero:", sum(tilde_D == 0))
            sqrt_tildD = 1. / torch.sqrt(tilde_D)
            daBda = lambda _b, _a: (_b * _a.transpose(1, 0)).transpose(1, 0) * _b
            hatA = daBda(sqrt_tildD, tilde_A)
        else:
            hatA = adj

        if W is None:
            W = self.W

        representation = F.relu(hatA @ x @ W[0], inplace=True)
        if self.training:
            representation = F.dropout(representation, self.dropout)

        for i in range(1, self.num_layer - 1):
            representation = F.relu(hatA @ representation @ W[i], inplace=True)
            if self.training:
                representation = F.dropout(representation, self.dropout)
        out = hatA @ representation @ W[-1]

        return out, W, representation

    def l2_loss(self, W=None):
        if W is None:
            W = self.W

        loss = W[0].pow(2).sum() / 2
        return loss

    def get_parameters(self):
        # this function used to get a copy of model parameter
        W_clone = []
        for p in self.W:
            W_clone.append(p.clone())
        return W_clone