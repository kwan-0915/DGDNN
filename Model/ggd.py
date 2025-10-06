"""Generalized graph diffusion operators."""

import torch
import torch.nn as nn
from torch import Tensor


class GeneralizedGraphDiffusion(nn.Module):
    """Compute diffused node embeddings over learned diffusion bases."""

    def __init__(self, input_dim: int, output_dim: int, active: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

    @staticmethod
    def _validate_inputs(
        theta: Tensor, T_slices: Tensor, x: Tensor, a: Tensor
    ) -> None:
        if theta.dim() != 1:
            raise ValueError("theta must be a 1-D tensor of diffusion coefficients")
        if T_slices.dim() != 3:
            raise ValueError("T_slices must be a 3-D tensor [S, N, N]")
        if x.dim() != 2:
            raise ValueError("x must be a 2-D tensor of node features")
        if a.dim() != 2 or a.size(0) != a.size(1):
            raise ValueError("Adjacency tensor a must be square")
        if T_slices.size(0) != theta.size(0):
            raise ValueError("Number of diffusion bases must match the coefficients")
        if T_slices.size(1) != a.size(0):
            raise ValueError("Diffusion bases and adjacency must share node dimension")
        if x.size(0) != a.size(0):
            raise ValueError("Node feature count must match adjacency size")

    def forward(
        self,
        theta: Tensor,
        T_slices: Tensor,
        x: Tensor,
        a: Tensor,
    ) -> Tensor:
        """Apply the generalized diffusion operator.

        Args:
            theta: ``[S]`` diffusion coefficients.
            T_slices: ``[S, N, N]`` diffusion bases.
            x: ``[N, F_in]`` node features.
            a: ``[N, N]`` adjacency weights.

        Returns:
            ``[N, F_out]`` tensor with diffused node representations.
        """

        self._validate_inputs(theta, T_slices, x, a)

        q = torch.einsum("s,sij->ij", theta, T_slices)
        q = q * a

        q_sparse = q.to_sparse().coalesce()
        out = torch.sparse.mm(q_sparse, x)

        out = self.activation(out)
        return self.fc(out)

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

