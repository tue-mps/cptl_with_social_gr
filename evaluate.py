import numpy as np
import torch
import utils
from data import data_loader


###---------------------------------------------------------------------------------------------------###

##-----------------------------##
##----PREDICTION EVALUATION----##
##-----------------------------##

def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade_ = utils.displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_ = utils.final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade_, fde_

def evaluate(args, loader, predictor):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) =batch

            total_traj += pred_traj_gt.size(1)
            pred_traj_fake = predictor(obs_traj)
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)

            ade_outer.append(ade_)
            fde_outer.append(fde_)
        ade = 0
        fde = 0
        for i in range(len(ade_outer)):
            ade += torch.sum(ade_outer[i]).item()
            fde += torch.sum(fde_outer[i]).item()
        ade = ade / (total_traj * args.pred_len)
        fde = fde / total_traj
        # a = sum(ade_outer)
        # b = total_traj * args.pred_len
        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)
        return ade, fde


def validate(args, model, dataset_name, batch_size=128, test_size=1024, verbose=True):
    '''
    Evaluate precision (ADE and FDE) of a predictor ([model]) on [dataset].
    '''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Loop over batches in [dataset]
    print("\nInitializing test dataset")
    test_path = utils.get_dset_path("test")
    data_type = ".txt"
    _, test_loader = data_loader(args, test_path, dataset_name, data_type)
    ade, fde = evaluate(args, test_loader, model)

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print("\nDataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            dataset_name, args.pred_len, ade, fde
        ))
    return ade, fde





