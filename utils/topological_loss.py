import torch
import torch.nn as nn

from torch_topological.nn.data import batch_iter

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

from torch_topological.utils import total_persistence

def topo_loss(pred_obj, true_obj, lamda, interp = 0,
            feat_d = 2, loss_q = 2, loss_r = False):
    """
    Calculate topological loss forward step.

    Args:
        pred_obj (torch.Tensor): object predicted by model in [B, C, D_1, ..., D_N] format
        true_obj (torch.Tensor): ground truth data in [B, C, D_1, ..., D_N] format
        lamda (float): strength of topological regularisation
        interp (int): size of downsampled input data (0 by default)
        feat_d (int):  dimension of topological features to use (2 by default)
        loss_q (int):  exponent for wasserstein loss calculations (2 by default)
        loss_r (boolean):  additional regularisation on topological information (False by default)
    
    Returns:
        loss (float): value of topological loss
    """

    # Check if it is necessary to do anything
    if lamda == 0.0:
        return 0.0

    if interp != 0:
        size = (interp,) * 2
        pred_obj_ = nn.functional.interpolate(
            input=pred_obj,
            size=size,
            mode='bilinear',
            align_corners=True,
        )
        true_obj_ = nn.functional.interpolate(
            input=true_obj,
            size=size,
            mode='bilinear',
            align_corners=True,
        )

    # No interpolation desired by client; use the original data set,
    # thus making everything slower.
    else:
        pred_obj_ = pred_obj
        true_obj_ = true_obj

    # Define cubical complex settings
    cubical_complex = CubicalComplex(
        dim = 2, 
        superlevel=False
    )
    # Calculate topological features of predicted 2D tensor and ground
    # truth 2D tensor. The 'squeeze()' ensures that single dimensions such
    # as channels or batches are ignored
    pers_info_pred = cubical_complex(pred_obj_.squeeze())
    pers_info_true = cubical_complex(true_obj_.squeeze())

    # Check whether all topological dimensional features must be used or
    # not. If `dim` is None we do not perform any filtering of the resulting
    # persistence information selfs
    dim = feat_d if 0 <= feat_d <= 2 else None
    if dim is not None:
        pers_info_pred = [
            x for x in batch_iter(pers_info_pred, dim = feat_d)
        ]
        pers_info_true = [
            x for x in batch_iter(pers_info_true, dim = feat_d)
        ]

    # Compute Wasserstein loss for every element of the batch
    wasserstein_dist = WassersteinDistance(q = loss_q)
    topological_loss = torch.stack([
        wasserstein_dist(pred_batch, true_batch)
        for pred_batch, true_batch in zip(pers_info_pred, pers_info_true)
    ])

    # Use a reduction method for the loss like mean
    topological_loss = topological_loss.mean()

    # In case `loss_r` is true, add regularization
    if loss_r:
        topo_reg = torch.stack([
            total_persistence(info.diagram, p = loss_q)
            for pred_batch in pers_info_pred for info in pred_batch
        ])

        topological_loss += topo_reg.mean()
    
    return lamda*topological_loss   