import numpy as np
import pickle
import os
import logging
import torch
from torch import nn
from torch.nn import functional as F
import copy


# Data-handling functions
def get_data_loader():
    pass

# Get dataset path
def get_dset_path(dset_type):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, "datasets", dset_type)

# relative to absolute
def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)




##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("    of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                   round(learnable_params / 1000000, 1)))
        print("              - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-")





# Calculate l2_loss
def l2_loss(pred_traj, pred_traj_gt, random=0, mode="average"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, size = pred_traj.size()
    # equation below , the first part do noing, can be delete

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / (seq_len*batch*size)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)

# Calculate ADE metric
def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    '''
    Input:
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth predictions.
    :param consider_ped: Tensor of shape (batch)
    :param mode: Can be one of sum, raw
    :return:
    - loss: gives the eculidian displacement error
    '''

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1,0,2) - pred_traj.permute(1,0,2)

    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) *consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

# Calculate FDE metric
def final_displacement_error(pred_pos, pred_pos_gt, condiser_ped=None, mode="sum"):
    '''
    Input:
    :param pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 2). Ground truth last pos.
    :param condiser_ped: Tensor of shape (batch).
    :param mode:
    :return: loss --> gives the eculidian displacement error
    '''

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if condiser_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * condiser_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)
