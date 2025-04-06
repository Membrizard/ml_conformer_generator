# """
# The Implementation is based on
# Grant, J.A., Pickup, B.T. (1997). Gaussian shape methods.
# In: van Gunsteren, W.F., Weiner, P.K., Wilkinson, A.J. (eds)
# Computer Simulation of Biomolecular Systems.
# Computer Simulations of Biomolecular Systems, vol 3. Springer, Dordrecht.
# https://doi.org/10.1007/978-94-017-1120-3_5
# """
from typing import List, Tuple, Union

import numpy as np
import torch

ATOM_RADIUS = 1.60
AMPLITUDE = 2.70


def get_shape_quadrupole_for_molecule(
    coordinates: torch.Tensor,
    amplitude: float = AMPLITUDE,
    generic_atom_radius: float = ATOM_RADIUS,
    n_terms: int = 6,
    neighbour_threshold: float = 2 * AMPLITUDE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates shape quadrupole for a molecule in a principal frame
    :param coordinates: atom coordinates
    :param amplitude: gaussian amplitude
    :param generic_atom_radius: generic radius of all atoms - 1.60 by default
    :param n_terms: maximal order of the products to include for shape descriptors computation
    :param neighbour_threshold: threshold to define neighbors
    :return: main elements of the principle shape quadrupole tensor,
             coordinates in a pricniple frame
    """
    alpha = get_alpha(atom_radius=generic_atom_radius, gaussian_amplitude=amplitude)

    volume = coordinates.size(0) * _0th_moment_integral(alpha, amplitude)

    first_moments = torch.sum(
        i_1st_moment_integral(
            center_coord_i=coordinates, alpha=alpha, amplitude=amplitude
        ),
        0,
    )

    combination_indices_dict = dict()

    for i in range(2, n_terms + 1):
        combination_indices = get_valid_combinations(
            coordinates=coordinates,
            neighbour_threshold=neighbour_threshold,
            subset_size=i,
        )

        # Store combinations for further calculations
        combination_indices_dict[i] = combination_indices
        combinations = coordinates[combination_indices]

        n_centers, n_alpha, n_amplitude = product_of_n_gaussians(
            centers=combinations, amplitude=amplitude, alpha=alpha
        )

        i_intersection_integral_matrix = _0th_moment_integral(
            alpha=n_alpha, amplitude=n_amplitude
        )
        intersection_volumes = (-1) ** (i - 1) * torch.sum(
            i_intersection_integral_matrix
        )

        i_intersection_first_moments = (-1) ** (i - 1) * torch.sum(
            i_1st_moment_integral(
                center_coord_i=n_centers,
                alpha=n_alpha,
                amplitude=n_amplitude.unsqueeze(-1),
            ),
            0,
        )
        volume += intersection_volumes
        first_moments += i_intersection_first_moments

    first_moments = first_moments / volume

    # Move the center of coordinates to First Moments
    centered_points = coordinates - first_moments

    ii_s_mom_0 = torch.sum(
        ii_2nd_moment_integral(
            center_coord_i=centered_points, alpha=alpha, amplitude=amplitude
        ),
        0,
    )

    ij_s_mom_0 = torch.sum(
        ij_2nd_moment_integral(
            center_coord=centered_points, alpha=alpha, amplitude=amplitude
        ),
        -1,
    )

    for i in range(2, n_terms + 1):
        combination_indices = combination_indices_dict[i]
        combinations = centered_points[combination_indices]

        n_centers, n_alpha, n_amplitude = product_of_n_gaussians(
            centers=combinations, amplitude=amplitude, alpha=alpha
        )

        ii_s_mom_0 += (-1) ** (i - 1) * torch.sum(
            ii_2nd_moment_integral(
                center_coord_i=n_centers,
                alpha=n_alpha,
                amplitude=n_amplitude.unsqueeze(-1),
            ),
            0,
        )

        ij_s_mom_0 += (-1) ** (i - 1) * torch.sum(
            ij_2nd_moment_integral(
                center_coord=n_centers,
                alpha=n_alpha,
                amplitude=n_amplitude.unsqueeze(0),
            ),
            -1,
        )

    s_mom_tensor_0 = (
        torch.tensor(
            [
                [ii_s_mom_0[0].item(), ij_s_mom_0[0].item(), ij_s_mom_0[1].item()],
                [ij_s_mom_0[0].item(), ii_s_mom_0[1].item(), ij_s_mom_0[2].item()],
                [ij_s_mom_0[1].item(), ij_s_mom_0[2].item(), ii_s_mom_0[2].item()],
            ]
        )
        / volume
    )

    # Rotate the molecule to set all non-main moments to zero

    _, eigenvectors = torch.linalg.eigh(s_mom_tensor_0)
    rotated_points = torch.matmul(centered_points, eigenvectors)

    ii_s_mom = torch.sum(
        ii_2nd_moment_integral(
            center_coord_i=rotated_points, alpha=alpha, amplitude=amplitude
        ),
        0,
    )

    ij_s_mom = torch.sum(
        ij_2nd_moment_integral(
            center_coord=rotated_points, alpha=alpha, amplitude=amplitude
        ),
        -1,
    )

    # Calculate main second moments after rotation

    for i in range(2, n_terms + 1):
        combination_indices = combination_indices_dict[i]
        combinations = rotated_points[combination_indices]

        n_centers, n_alpha, n_amplitude = product_of_n_gaussians(
            centers=combinations, amplitude=amplitude, alpha=alpha
        )

        ii_s_mom += (-1) ** (i - 1) * torch.sum(
            ii_2nd_moment_integral(
                center_coord_i=n_centers,
                alpha=n_alpha,
                amplitude=n_amplitude.unsqueeze(-1),
            ),
            0,
        )

        ij_s_mom += (-1) ** (i - 1) * torch.sum(
            ij_2nd_moment_integral(
                center_coord=n_centers,
                alpha=n_alpha,
                amplitude=n_amplitude.unsqueeze(0),
            ),
            -1,
        )

    s_mom_tensor = (
        torch.tensor(
            [
                [ii_s_mom[0].item(), ij_s_mom[0].item(), ij_s_mom[1].item()],
                [ij_s_mom[0].item(), ii_s_mom[1].item(), ij_s_mom[2].item()],
                [ij_s_mom[1].item(), ij_s_mom[2].item(), ii_s_mom[2].item()],
            ]
        )
        / volume
    )

    # Set coordinates in a way, that XX is the largest moment

    main_moments = torch.diag(s_mom_tensor)
    final_moments, indices = torch.sort(main_moments, descending=True)

    final_points = rotated_points[:, indices]

    return final_moments, final_points


def product_of_n_gaussians(
    centers: torch.Tensor,
    alpha: float,
    amplitude: float = AMPLITUDE,
) -> Tuple[torch.Tensor, Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculates product of n Gaussians within a batch with the same amplitude and alpha, but different centers.
    :param centers: list of centers of Gaussians to be multiplied (n, subset_size, 3)
    :param alpha: gaussian alpha
    :param amplitude: gaussian amplitude
    :return: batch of new_center coordinates, new_alpha as float, batch of new amplitudes
    """
    n = centers.size(1)

    new_centers = torch.mean(centers, 1)

    r2_sum = torch.sum(torch.sum(torch.pow(centers, 2), -1), -1)

    xyz_k_sum = torch.sum(torch.pow(torch.sum(centers, 1), 2), -1) / n

    gamma = r2_sum - xyz_k_sum

    new_amplitude = amplitude**n * torch.exp(-alpha * gamma)
    new_alpha = n * alpha

    return new_centers, new_alpha, new_amplitude


def get_valid_combinations(
    coordinates: torch.Tensor, neighbour_threshold: float, subset_size: int
) -> torch.Tensor:
    """
    Get valid combinations of gaussians for shape descriptors
    :param coordinates: set of atom coordinates
    :param neighbour_threshold: a threshold to define neighbours
    :param subset_size: size of the combination
    :return: a tensor with all valid combinations of indices of input coordinates,
             such as all elements within each combination are mutual neighbors
    """

    n = coordinates.size(0)
    i_mat = coordinates.unsqueeze(1).repeat(
        1, n, 1
    )  # Repeat coordinates tensor along new dimension
    j_mat = i_mat.transpose(0, 1)

    # Create a distance matrix
    dist_mat = torch.sqrt(
        torch.sum(torch.pow(i_mat - j_mat, 2), 2)
    )  # => Add self connections

    # Nullify all values which are greater than R boundary
    dist_mat[dist_mat >= neighbour_threshold] = 0
    # Set all non-nullified values to 1 - > Somewhat of an adjacency matrix
    dist_mat[dist_mat > 0] = 1

    # Get all combinations of mutually adjacent nodes (cliques of rank n)
    valid_combination_indices = find_r_cliques_fast(
        adj_mat=dist_mat, clique_order=subset_size
    )

    return valid_combination_indices


def find_r_cliques_fast(adj_mat: torch.Tensor, clique_order: int) -> torch.Tensor:
    """
    Find all r-cliques in a boolean adjacency matrix using a neighbor-intersection approach
    :param adj_mat: (n,n) bool adjacency
    :param clique_order: size of the cliques
    :return:A (C, r) tensor of node indices, each row is an r-clique.
    """
    n = adj_mat.size(0)
    neigh_masks = build_neighbor_sets(adj_mat)

    cliques_found = []

    def backtrack(partial_clique, candidates_mask, start_idx):
        if len(partial_clique) == clique_order:
            cliques_found.append(partial_clique[:])
            return

        # Pruning: if not enough candidates to reach r
        needed = clique_order - len(partial_clique)
        if candidates_mask.sum().item() < needed:
            return

        cands_idx = candidates_mask.nonzero(as_tuple=True)[0]
        cands_idx = cands_idx[cands_idx >= start_idx]

        for node in cands_idx:
            node_int = node.item()
            partial_clique.append(node_int)

            next_candidates = torch.logical_and(candidates_mask, neigh_masks[node_int])
            next_candidates[:node_int] = False

            backtrack(partial_clique, next_candidates, node_int + 1)

            partial_clique.pop()

    full_mask = torch.ones(n, dtype=torch.bool, device=adj_mat.device)
    backtrack(partial_clique=[], candidates_mask=full_mask, start_idx=0)

    if len(cliques_found) == 0:
        return torch.empty((0, clique_order), dtype=torch.long)

    return torch.tensor(cliques_found, dtype=torch.long)


def build_neighbor_sets(adj_mat: torch.Tensor) -> List:
    """
    Converts (n,n) bool adjacency into a list of boolean masks for quick neighbor intersection.

    Returns:
      neigh_masks: a list of length n, where neigh_masks[i] is a bool tensor
                   of shape (n,) with True at neighbors of i.
    """
    n = adj_mat.size(0)
    neigh_masks = [adj_mat[i] for i in range(n)]
    return neigh_masks


def get_alpha(
    atom_radius: float = ATOM_RADIUS, gaussian_amplitude: float = AMPLITUDE
) -> float:
    # Calculate alpha
    lyambda_ = 4 * np.pi / 3 / gaussian_amplitude
    k_a = np.pi / lyambda_ ** (2 / 3)
    alpha = k_a / atom_radius**2
    return alpha


def _0th_moment_integral(
    alpha: float, amplitude: torch.Tensor or float
) -> torch.Tensor:
    """Calculate the integral of a 3D Gaussian.
    f(x) = A·exp(-alpha * (|x-c_1|²+|y-c_2|²+|z-c_3|²))
    The integral over all of R³ is:
    ∫f(x)dx = A·(π / alpha)^(3/2)
    """

    return amplitude * (np.pi / alpha) ** (3 / 2)


def i_1st_moment_integral(
    center_coord_i: torch.Tensor, alpha: float, amplitude: torch.Tensor or float
) -> torch.Tensor:
    """
    Calculate the integral of a 3D Gaussian.
    f(x) = A·x·exp(-alpha · (|x-c_1|²+|y-c_2|²+|z-c_3|²))
    The integral over all of R³ is:
    ∫f(x)dx = A·c_i·(π / alpha)^(3/2)
    """

    return amplitude * center_coord_i * (np.pi / alpha) ** (3 / 2)


def ii_2nd_moment_integral(
    center_coord_i: torch.Tensor, alpha: float, amplitude: torch.Tensor or float
) -> torch.Tensor:
    """
    Calculate the integral of a 3D Gaussian.
    f(x) = A·x·exp(-alpha · (|x-c_1|²+|y-c_2|²+|z-c_3|²))
    The integral over all of R³ is:
    ∫f(x)dx = A·(π / alpha)^(3/2)·(c_i² - 1/(2 · alpha))
    """

    return (
        amplitude
        * (np.pi / alpha) ** (3 / 2)
        * (torch.pow(center_coord_i, 2) + 1 / (2 * alpha))
    )


def ij_2nd_moment_integral(
    center_coord: torch.Tensor, alpha: float, amplitude: torch.Tensor or float
) -> torch.Tensor:
    """
    Calculate the integral of a 3D Gaussian.
    f(x) = A·x·exp(-alpha · (|x-c_1|²+|y-c_2|²+|z-c_3|²))
    The integral over all of R³ is:
    ∫f(x)dx = A·c_i·c_j·(π / alpha)^(3/2)
    """

    paired_coord_products = torch.stack(
        (
            center_coord[:, 0] * center_coord[:, 1],  # -> xy
            center_coord[:, 0] * center_coord[:, 2],  # -> xz
            center_coord[:, 1] * center_coord[:, 2],  # -> yz
        ),
        0,
    )

    return amplitude * paired_coord_products * (np.pi / alpha) ** (3 / 2)


# --------------------------------------------------------------------------
# Tanimoto score


class Grid:
    def __init__(
        self,
        min_coords,
        max_coords,
        bounds_scale: float = 6,
        max_sigma: float = ATOM_RADIUS,
        n: int = 4,
    ):
        min_coords = min_coords - bounds_scale * max_sigma
        max_coords = max_coords + bounds_scale * max_sigma

        xs = torch.linspace(min_coords[0], max_coords[0], n)
        ys = torch.linspace(min_coords[1], max_coords[1], n)
        zs = torch.linspace(min_coords[2], max_coords[2], n)

        x_g, y_g, z_g = torch.meshgrid(xs, ys, zs, indexing="ij")

        # Riemann sum
        dx = (max_coords[0] - min_coords[0]) / (n - 1)
        dy = (max_coords[1] - min_coords[1]) / (n - 1)
        dz = (max_coords[2] - min_coords[2]) / (n - 1)
        dV = dx * dy * dz

        self.points = torch.stack([x_g.flatten(), y_g.flatten(), z_g.flatten()], dim=-1)
        self.d_v = dV
        self.size = n


def torch_evaluate_density_on_grid(
    coordinates,
    grid: Grid,
    alpha: float,
    amplitude: float = AMPLITUDE,
) -> torch.Tensor:
    grid_points = grid.points
    dist_sq = torch.cdist(grid_points, coordinates) ** 2
    gaussian_vals = amplitude * torch.exp(-dist_sq * alpha)
    density = 1 - torch.prod(1 - gaussian_vals, dim=-1)

    return density


def rotate_coord(coord: torch.Tensor, angles: torch.Tensor):
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Rotation matrices

    rot_x = torch.tensor([[1, 0, 0], [0, cos_a[0], -sin_a[0]], [0, sin_a[0], cos_a[0]]])
    rot_y = torch.tensor([[cos_a[1], 0, sin_a[1]], [0, 1, 0], [-sin_a[1], 0, cos_a[1]]])
    rot_z = torch.tensor([[cos_a[2], -sin_a[2], 0], [sin_a[2], cos_a[2], 0], [0, 0, 1]])

    # Rotate the structure around the origin

    out = torch.matmul(torch.matmul(torch.matmul(coord, rot_x), rot_y), rot_z)
    return out


ALPHA = get_alpha(atom_radius=ATOM_RADIUS, gaussian_amplitude=AMPLITUDE)


# Implementation of Gaussian volumes intersection tanimoto score
def tanimoto_score(
    ref_coord,
    cand_coord,
    alpha: float = ALPHA,
    amplitude: float = AMPLITUDE,
    n: int = 40,
):
    # alpha = get_alpha(atom_radius=atom_radius, gaussian_amplitude=amplitude)

    cat_coord = torch.cat((ref_coord, cand_coord), dim=0)

    min_, _ = torch.min(cat_coord, dim=1)
    max_, _ = torch.max(cat_coord, dim=1)
    grid = Grid(min_coords=min_, max_coords=max_, n=n)

    ref_density = torch_evaluate_density_on_grid(ref_coord, grid, alpha, amplitude)

    cand_density = torch_evaluate_density_on_grid(cand_coord, grid, alpha, amplitude)

    f_g_integral = torch.sum(ref_density * cand_density)
    f_2 = torch.sum(ref_density * ref_density)
    g_2 = torch.sum(cand_density * cand_density)

    score = f_g_integral / (f_2 + g_2 - f_g_integral)

    return score.item()
