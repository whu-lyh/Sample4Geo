
import math

import torch
import torch.nn.functional as F
from torch import nn


class GatingContext(nn.Module):
    """
        Dimension reduction? # TODO
    """
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates
        return activation


class NetVLADLoupe(nn.Module):
    """
        NetVLAD layer
        From paper PointNetVLAD
    """
    def __init__(self, feature_size, num_clusters, output_dim, 
                 gating=True, add_batch_norm=True, work_with_tokens=True):
        """
        Args:
            num_clusters : int, by default 64
                The number of clusters.
            feature_size : int, by default 256
                Dimension of input local descriptors.
            output_dim : int, by default 256
                Dimension of output descriptors.
            add_batch_norm : bool, by default False
                Do batch normalization
            gating : bool, by default True
                If true, descriptor-wise L2 normalization is applied to input.
            work_with_tokens: bool, by default True
                If true, the backbone is from vit model
        """
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.num_clusters = num_clusters
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.work_with_tokens = work_with_tokens # True if backbone is vit-based

        self.softmax = nn.Softmax(dim=-1)
        # make the vlad differentiable
        self.cluster_weights = nn.Parameter(torch.randn(self.feature_size, self.num_clusters) * 1 / math.sqrt(self.feature_size)) # C K
        # VLAD Core centroids that will be learnable parameters
        self.cluster_weights2 = nn.Parameter(torch.randn(1, self.feature_size, self.num_clusters) * 1 / math.sqrt(self.feature_size)) # 1 C K
        # GatingContext
        self.hidden1_weights = nn.Parameter(torch.randn(self.num_clusters * self.feature_size, self.output_dim) * 1 / math.sqrt(self.feature_size)) # K*C, output dimension

        if self.add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(self.num_clusters)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(self.num_clusters) * 1 / math.sqrt(self.feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(self.output_dim)
        if self.gating:
            self.context_gating = GatingContext(self.output_dim, add_batch_norm=self.add_batch_norm)

    def forward(self, x, token_weight=None):
        """NetVLAD forward function

        Args:
            x (torch.Tensor): input feature, size = BNC or BCN1
            token_weight (torch.Tensor, optional): token weights from attention. size = B, N, Defaults to None.

        Returns:
            vlad global feature (torch.Tensor): size = B D
        """
        if self.work_with_tokens: # from transformer token output
            x = x.permute(0, 2, 1).unsqueeze(3) # BNC -> BCN -> BCN1 # for be compatible with netvlad layer
        # make sure input x == B C N 1
        self.max_samples = x.shape[2]
        x = x.transpose(1, 3).contiguous() # B C N 1 -> B 1 N C
        x = x.view((-1, self.max_samples, self.feature_size)) # B 1 N C -> B*1 N C = B N C
        # a learnable parameter that indicate the confidience of each local belonging to each cluster
        activation = torch.matmul(x, self.cluster_weights) # B N C * C K = B N K

        if self.add_batch_norm:
            activation = activation.view(-1, self.num_clusters) # B N K -> B*N K
            activation = self.bn1(activation) # B*N K
            activation = activation.view(-1, self.max_samples, self.num_clusters) # B*N K -> B N K
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation) # B N K
        if token_weight is not None:
            # poor performance
            # token_weight = F.normalize(token_weight, dim=-1)
            # activation = activation * token_weight.unsqueeze(-1) # B N K * B N = B N K
            # better performance
            token_weight = token_weight.reshape([-1, self.max_samples, 1])
            token_weight = torch.sigmoid(token_weight)
            token_weight = token_weight.repeat(1, 1, self.num_clusters)
            activation = torch.mul(activation, token_weight)

        activation = activation.view((-1, self.max_samples, self.num_clusters)) # B N K -> B N K
        # sum from whole N local features
        a_sum = activation.sum(-2, keepdim=True) # B 1 K
        a = a_sum * self.cluster_weights2 # B 1 K * 1 C K = B C K
        # same results between * and torch.mul # element-wise multiply, broadcast mechanism
        # a = torch.mul(a_sum, self.cluster_weights2)

        activation = torch.transpose(activation, 2, 1) # B N K -> B K N
        x = x.view((-1, self.max_samples, self.feature_size)) # B N C -> B N C
        vlad = torch.matmul(activation, x) # B K N * B N C = B K C
        vlad = torch.transpose(vlad, 2, 1) # B K C -> B C K
        vlad = vlad - a # B C K - B C K = B C K

        vlad = F.normalize(vlad, dim=1, p=2) # intra-normalization, inside cluster # B C K
        vlad = vlad.contiguous().view((-1, self.num_clusters * self.feature_size)) # flatten vlad # B C K -> B C*K
        vlad = F.normalize(vlad, dim=1, p=2) # L2-normalization # B C*K
        vlad = torch.matmul(vlad, self.hidden1_weights) # B C*K * C*K C_OUT = B C_OUT
        vlad = self.bn2(vlad)
        if self.gating:
            vlad = self.context_gating(vlad)
        return vlad

    def forward_vlad_residual(self, x, token_weight=None):
        """NetVLAD forward function that return the residual feature

        Args:
            x (torch.Tensor): input feature, size = BNC or BCN1
            token_weight (torch.Tensor, optional): token weights from attention. size = B, N, Defaults to None.

        Returns:
            vlad residual feature (torch.Tensor): size = B D K
        """
        if self.work_with_tokens: # from transformer token output
            x = x.permute(0, 2, 1).unsqueeze(3) # BNC -> BCN -> BCN1 # for be compatible with netvlad layer
        # make sure input x == B C N 1
        self.max_samples = x.shape[2]
        x = x.transpose(1, 3).contiguous() # B C N 1 -> B 1 N C
        x = x.view((-1, self.max_samples, self.feature_size)) # B 1 N C -> B*1 N C = B N C
        # a learnable parameter that indicate the confidience of each local belonging to each cluster
        activation = torch.matmul(x, self.cluster_weights) # B N C * C K = B N K

        if self.add_batch_norm:
            activation = activation.view(-1, self.num_clusters) # B N K -> B*N K
            activation = self.bn1(activation) # B*N K
            activation = activation.view(-1, self.max_samples, self.num_clusters) # B*N K -> B N K
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation) # B N K
        if token_weight is not None:
            # poor performance
            # token_weight = F.normalize(token_weight, dim=-1)
            # activation = activation * token_weight.unsqueeze(-1) # B N K * B N = B N K
            # better performance
            token_weight = token_weight.reshape([-1, self.max_samples, 1])
            token_weight = torch.sigmoid(token_weight)
            token_weight = token_weight.repeat(1, 1, self.num_clusters)
            activation = torch.mul(activation, token_weight)

        activation = activation.view((-1, self.max_samples, self.num_clusters)) # B N K -> B N K
        # sum from whole N local features
        a_sum = activation.sum(-2, keepdim=True) # B 1 K
        a = a_sum * self.cluster_weights2 # B 1 K * 1 C K = B C K
        # same results between * and torch.mul # element-wise multiply, broadcast mechanism
        # a = torch.mul(a_sum, self.cluster_weights2)

        activation = torch.transpose(activation, 2, 1) # B N K -> B K N
        x = x.view((-1, self.max_samples, self.feature_size)) # B N C -> B N C
        vlad = torch.matmul(activation, x) # B K N * B N C = B K C
        vlad = torch.transpose(vlad, 2, 1) # B K C -> B C K
        vlad = vlad - a # B C K - B C K = B C K
        vlad = F.normalize(vlad, dim=1, p=2) # intra-normalization, inside cluster # B C K
        return vlad
