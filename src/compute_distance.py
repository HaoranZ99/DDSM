from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import eigh
import scipy.sparse as sp
import sys
import numpy as np
import torch


def compute_distance(
    adj, deg, edge_index, t, k, seed, gamma, diffusion_type="pagerank"
):
    if diffusion_type in ["vanilla", "heat", "pagerank"]:
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
        deg_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_tilde = deg_inv_sqrt @ adj @ deg_inv_sqrt
        # Eigendecomposion.
        if k == -1:
            k = adj.shape[0]
        if diffusion_type in ["vanilla"]:
            u, sigma, _ = randomized_svd(
                adj_tilde, n_components=k, n_iter=10, random_state=seed
            )
        elif diffusion_type in ["pagerank", "heat"]:
            if k == adj.shape[0]:
                sigma, u = eigh(adj_tilde.toarray())
            else:
                sigma, u = eigsh(adj_tilde, k=k, which="LA")
            sorted_indices = np.argsort(-sigma)
            sigma = sigma[sorted_indices]
            u = u[:, sorted_indices]
            sigma[sigma > 1] = 1
            if diffusion_type == "heat":
                sigma = 1 - sigma
        sigma = sp.diags(sigma)
        if diffusion_type == "vanilla":
            distance = deg_inv_sqrt @ u @ np.power(sigma, t)
        elif diffusion_type == "heat":
            # https://cran.r-project.org/web/packages/diffudist/vignettes/diffudist-package.html.
            distance = deg_inv_sqrt @ u @ sp.linalg.expm(-t * sigma.tocsc())
        elif diffusion_type == "pagerank":
            distance = (
                deg_inv_sqrt
                @ u
                @ sp.linalg.inv((sp.eye(sigma.shape[0]) - gamma * sigma).tocsc())
            )
        distance = torch.Tensor(distance)
        distance = torch.norm(
            distance[edge_index[0]] - distance[edge_index[1]], p=2, dim=1
        )
    else:
        sys.exit("Unknown Diffusion Type!")
    return distance
