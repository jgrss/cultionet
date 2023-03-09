import typing as T

import numpy as np
import torch
import gudhi


def critical_points(
    x: torch.Tensor,
) -> T.Tuple[T.List[np.ndarray], T.List[np.ndarray], T.List[np.ndarray], bool]:
    batch_size = x.shape[0]
    lh_vector = 1.0 - x.flatten()
    cubical_complex = gudhi.CubicalComplex(
        dimensions=x.shape, top_dimensional_cells=lh_vector
    )
    cubical_complex.persistence(homology_coeff_field=2, min_persistence=0)
    cofaces = cubical_complex.cofaces_of_persistence_pairs()
    cofaces_batch_size = len(cofaces[0])

    if (cofaces_batch_size == 0) or (cofaces_batch_size != batch_size):
        return None, None, None, False

    pd_lh = [
        np.c_[
            lh_vector[cofaces[0][batch][:, 0]],
            lh_vector[cofaces[0][batch][:, 1]],
        ]
        for batch in range(0, batch_size)
    ]
    bcp_lh = [
        np.c_[
            cofaces[0][batch][:, 0] // x.shape[-1],
            cofaces[0][batch][:, 0] % x.shape[-1],
        ]
        for batch in range(0, batch_size)
    ]
    dcp_lh = [
        np.c_[
            cofaces[0][batch][:, 1] // x.shape[-1],
            cofaces[0][batch][:, 1] % x.shape[-1],
        ]
        for batch in range(0, batch_size)
    ]

    return pd_lh, bcp_lh, dcp_lh, True


def compute_dgm_force(
    lh_dgm: np.ndarray,
    gt_dgm: np.ndarray,
    pers_thresh: float = 0.03,
    pers_thresh_perfect: float = 0.99,
    do_return_perfect: bool = False,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Compute the persistent diagram of the image.

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
            between the local maximum critical point value with its neighouboring minimum critical point value.
            Values smaller than the persistent threshold should be filtered. Default is 0.03.
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
            correct match. Default is 0.99.
        do_return_perfect: Return the persistent point or not from the matching. Default is ``False``.

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram.
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process.
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
            process.
    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if gt_dgm.shape[0] == 0:
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None) or (gt_n_holes == 0):
        idx_holes_to_fix = np.array([], dtype=int)
        idx_holes_to_remove = np.array(list(set(range(lh_pers.size))))
        idx_holes_perfect = []
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = []

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = np.array(
            list(set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))
        )

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = np.array(
        list(set(idx_holes_to_remove).intersection(set(idx_valid)))
    )

    force_list = np.zeros(lh_dgm.shape)

    # push each hole-to-fix to (0,1)
    if idx_holes_to_fix.shape[0] > 0:
        force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
        force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    if idx_holes_to_remove.shape[0] > 0:
        force_list[idx_holes_to_remove, 0] = lh_pers[
            idx_holes_to_remove
        ] / np.sqrt(2.0)
        force_list[idx_holes_to_remove, 1] = -lh_pers[
            idx_holes_to_remove
        ] / np.sqrt(2.0)

    if do_return_perfect:
        return (
            force_list,
            idx_holes_to_fix,
            idx_holes_to_remove,
            idx_holes_perfect,
        )

    return force_list, idx_holes_to_fix, idx_holes_to_remove


def adjust_holes_to_fix(
    topo_cp_weight_map: np.ndarray,
    topo_cp_ref_map: np.ndarray,
    topo_mask: np.ndarray,
    hole_indices: np.ndarray,
    pairs: np.ndarray,
    fill_weight: int,
    fill_ref: int,
    height: int,
    width: int,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        (pairs[hole_indices][:, 0] >= 0)
        * (pairs[hole_indices][:, 0] < height)
        * (pairs[hole_indices][:, 1] >= 0)
        * (pairs[hole_indices][:, 1] < width)
    )
    indices = (
        pairs[hole_indices][:, 0][mask],
        pairs[hole_indices][:, 1][mask],
    )
    topo_cp_weight_map[indices] = fill_weight
    topo_cp_ref_map[indices] = fill_ref
    topo_mask[indices] = 1

    return topo_cp_weight_map, topo_cp_ref_map, topo_mask


def adjust_holes_to_remove(
    likelihood: np.ndarray,
    topo_cp_weight_map: np.ndarray,
    topo_cp_ref_map: np.ndarray,
    topo_mask: np.ndarray,
    hole_indices: np.ndarray,
    pairs_b: np.ndarray,
    pairs_d: np.ndarray,
    fill_weight: int,
    fill_ref: int,
    height: int,
    width: int,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        (pairs_b[hole_indices][:, 0] >= 0)
        * (pairs_b[hole_indices][:, 0] < height)
        * (pairs_b[hole_indices][:, 1] >= 0)
        * (pairs_b[hole_indices][:, 1] < width)
    )
    indices = (
        pairs_b[hole_indices][:, 0][mask],
        pairs_b[hole_indices][:, 1][mask],
    )
    topo_cp_weight_map[indices] = fill_weight
    topo_mask[indices] = 1

    nested_mask = (
        mask
        * (pairs_d[hole_indices][:, 0] >= 0)
        * (pairs_d[hole_indices][:, 0] < height)
        * (pairs_d[hole_indices][:, 1] >= 0)
        * (pairs_d[hole_indices][:, 1] < width)
    )
    indices_b = (
        pairs_b[hole_indices][:, 0][nested_mask],
        pairs_b[hole_indices][:, 1][nested_mask],
    )
    indices_d = (
        pairs_d[hole_indices][:, 0][nested_mask],
        pairs_d[hole_indices][:, 1][nested_mask],
    )
    topo_cp_ref_map[indices_b] = likelihood[indices_d]
    topo_mask[indices_b] = 1

    indices_inv = (
        pairs_b[hole_indices][:, 0][mask],
        pairs_b[hole_indices][:, 1][mask],
    )
    topo_cp_ref_map[indices_inv] = fill_ref
    topo_mask[indices_inv] = 1

    return topo_cp_weight_map, topo_cp_ref_map, topo_mask


def set_topology_weights(
    likelihood: np.ndarray,
    topo_cp_weight_map: np.ndarray,
    topo_cp_ref_map: np.ndarray,
    topo_mask: np.ndarray,
    bcp_lh: np.ndarray,
    dcp_lh: np.ndarray,
    idx_holes_to_fix: np.ndarray,
    idx_holes_to_remove: np.ndarray,
    height: int,
    width: int,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = 0
    y = 0

    if len(idx_holes_to_fix) > 0:
        topo_cp_weight_map, topo_cp_ref_map, topo_mask = adjust_holes_to_fix(
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask=topo_mask,
            hole_indices=idx_holes_to_fix,
            pairs=bcp_lh,
            fill_weight=1,
            fill_ref=0,
            height=height,
            width=width,
        )
        topo_cp_weight_map, topo_cp_ref_map, topo_mask = adjust_holes_to_fix(
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask=topo_mask,
            hole_indices=idx_holes_to_fix,
            pairs=dcp_lh,
            fill_weight=1,
            fill_ref=1,
            height=height,
            width=width,
        )
    if len(idx_holes_to_remove) > 0:
        (
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask,
        ) = adjust_holes_to_remove(
            likelihood,
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask=topo_mask,
            hole_indices=idx_holes_to_remove,
            pairs_b=bcp_lh,
            pairs_d=dcp_lh,
            fill_weight=1,
            fill_ref=1,
            height=height,
            width=width,
        )
        (
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask,
        ) = adjust_holes_to_remove(
            likelihood,
            topo_cp_weight_map,
            topo_cp_ref_map,
            topo_mask=topo_mask,
            hole_indices=idx_holes_to_remove,
            pairs_b=dcp_lh,
            pairs_d=bcp_lh,
            fill_weight=1,
            fill_ref=0,
            height=height,
            width=width,
        )

    return topo_cp_weight_map, topo_cp_ref_map, topo_mask
