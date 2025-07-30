import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing, Linear, MLP, GCNConv
from torch_geometric.utils import degree, to_scipy_sparse_matrix


class DDSMConv(MessagePassing):
    def __init__(self, channels, alpha, beta, eta, gamma, eps=1e-5):
        super(DDSMConv, self).__init__(aggr="add")
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.eps = eps
        self.channels = channels

        self.linear = Linear(
            in_channels=channels,
            out_channels=channels,
            bias=True,
            weight_initializer="glorot",
        )

    def forward(
        self, x, x_0, edge_index, diffusion_distance, norm, norm_ii, deg_inv_sqrt
    ):
        x_norm = F.normalize(x, p=2, dim=0)  # Feature Normalization.
        orthogonal_message = x_norm @ (x_norm.t() @ x_norm)
        x = (
            self.alpha * x_0
            - self.beta * orthogonal_message
            + self.propagate(
                x=x,
                edge_index=edge_index,
                diffusion_distance=diffusion_distance,
                norm=norm,
                norm_ii=norm_ii,
                deg_inv_sqrt=deg_inv_sqrt.reshape(-1, 1),
            )
        )
        return x

    def message(
        self,
        x_i,
        x_j,
        diffusion_distance,
        norm,
        norm_ii,
        deg_inv_sqrt_i,
        deg_inv_sqrt_j,
    ):
        topological_message = norm.view(-1, 1) * x_j
        positional_message = (
            self.eta
            * diffusion_distance.view(-1, 1)
            * (norm_ii.view(-1, 1) * x_i - norm.view(-1, 1) * x_j)
            / (
                (
                    torch.norm(
                        (
                            deg_inv_sqrt_i.view(-1, 1) * x_i
                            - deg_inv_sqrt_j.view(-1, 1) * x_j
                        ),
                        p=2,
                        dim=1,
                    )
                    + self.eps
                ).view(-1, 1)
            )
        )
        return (1 - self.alpha + self.beta) * (topological_message + positional_message)


class DDSM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        alpha,
        beta,
        eta,
        gamma,
        dropout,
        eps=1e-5,
    ):
        super(DDSM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.dropout = dropout
        self.eps = eps

        self.initial = Linear(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            bias=True,
            weight_initializer="glorot",
        )

        self.convs = ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                DDSMConv(
                    channels=self.hidden_channels,
                    alpha=self.alpha,
                    beta=self.beta,
                    eta=self.eta,
                    gamma=self.gamma,
                )
            )
        self.final = Linear(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            bias=True,
            weight_initializer="glorot",
        )

        self.norm_cache = None
        self.norm_ii_cache = None
        self.deg_inv_sqrt_cache = None
        self.distance_cache = None

    def forward(self, x, edge_index, diffusion_distance):
        if self.norm_cache is None:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            self.norm_cache = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            self.norm_ii_cache = deg_inv_sqrt[row] * deg_inv_sqrt[row]
            self.deg_inv_sqrt_cache = deg_inv_sqrt
            self.distance_cache = diffusion_distance
        # Embedding.
        x = self.initial(x)
        x_0 = x
        # Get distance.
        diffusion_distance = self.distance_cache
        # Graph Propagation.
        for i in range(self.num_layers):
            x = self.convs[i].forward(
                x=x,
                x_0=x_0,
                edge_index=edge_index,
                diffusion_distance=diffusion_distance,
                norm=self.norm_cache,
                norm_ii=self.norm_ii_cache,
                deg_inv_sqrt=self.deg_inv_sqrt_cache,
            )
            x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
            x = F.relu(x, inplace=True)
        # Classification.
        x = self.final(x)
        return x
