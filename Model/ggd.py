"""Generalized graph diffusion operators."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["GeneralizedGraphDiffusion"]


class GeneralizedGraphDiffusion(nn.Module):
    """Compute diffused node embeddings over learned diffusion bases."""

    def __init__(self, input_dim: int, output_dim: int, active: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    @staticmethod
    def _validate_inputs(theta: Tensor, bases: Tensor, features: Tensor, adjacency: Tensor) -> None:
        if theta.dim() != 1:
            raise ValueError("theta must be a 1-D tensor of diffusion coefficients")
        if bases.dim() != 3:
            raise ValueError("bases must be a 3-D tensor with shape [S, N, N]")
        if features.dim() != 2:
            raise ValueError("features must be a 2-D tensor of node embeddings")
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must be a square matrix")
        if bases.size(0) != theta.size(0):
            raise ValueError("number of diffusion bases must match theta coefficients")
        if bases.size(1) != adjacency.size(0):
            raise ValueError("diffusion bases and adjacency must share node dimension")
        if features.size(0) != adjacency.size(0):
            raise ValueError("node feature count must equal adjacency size")

    def forward(self, theta: Tensor, bases: Tensor, features: Tensor, adjacency: Tensor) -> Tensor:
        """Apply the generalized diffusion operator."""

        self._validate_inputs(theta, bases, features, adjacency)

        device = features.device
        if adjacency.device != device:
            raise ValueError("features and adjacency must reside on the same device")
        if bases.device != device or theta.device != device:
            raise ValueError("Diffusion parameters must reside on the same device as inputs")

        diffusion_kernel = torch.einsum("s,sij->ij", theta, bases)

        if adjacency.layout == torch.strided:
            adjacency_matrix = adjacency
        else:
            adjacency_matrix = adjacency.to_dense()

        dtype = features.dtype
        diffusion_kernel = diffusion_kernel.to(dtype)
        adjacency_matrix = adjacency_matrix.to(dtype)
        diffused = (diffusion_kernel * adjacency_matrix) @ features.to(dtype)
        diffused = diffused.to(dtype)

        activated = self.activation(diffused)
        return self.fc(activated)

# Fast approximation version using GCNConv for efficiency
# from torch_geometric.nn import GCNConv
# import torch.nn as nn
# import torch

# class GeneralizedGraphDiffusion(nn.Module):
#     def __init__(self, input_dim, output_dim, active: bool):
#         super().__init__()
#         self.gconv = GCNConv(input_dim, output_dim)
#         self.activation = nn.PReLU(num_parameters=output_dim) if active else nn.Identity()

#     def forward(
#         self,
#         x: torch.Tensor,                  # [N, F_in]
#         edge_index: torch.Tensor,        # [2, E] COO format
#         edge_weight: torch.Tensor        # [E] edge weights (like q_ij values)
#     ) -> torch.Tensor:
#         x = self.gconv(x, edge_index, edge_weight)  # [N, F_out]
#         x = self.activation(x)
#         return x

