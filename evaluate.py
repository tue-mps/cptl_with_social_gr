import numpy as np
import torch
import utils
import visual_visdom
from data import data_loader


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_



####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####

def initiate_metrics_dict(n_tasks):
    metrics_dict = {}
    metrics_dict["average_ade"] = []
    metrics_dict["average_fde"] = []
    metrics_dict["x_iteration"] = []
    metrics_dict["x_task"] = []
    metrics_dict["ade per task"] = {}
    metrics_dict["fde per task"] = {}
    for i in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(i+1)] = []
        metrics_dict["fde per task"]["task {}".format(i+1)] = []
    return metrics_dict

def intial_accuracy(model, datasets, metric_dict, test_size=None, verbose=False, no_task_mask=False):
    n_tasks = len(datasets)
    ades = []
    fdes = []

    for i in range(n_tasks):
        ade, fde = validate(model, datasets[i])
        ades.append(ade)
        fdes.append(fde)

    metric_dict["initial ade per task"] = ades
    metric_dict["initial fde per task"] = fdes
    return metric_dict


def metric_statistics(model, datasets, current_task, iteration,
                      metrics_dict=None, test_size=None, verbose=False):
    n_tasks = len(datasets)
    ades_all_classes = []
    fdes_all_classes = []
    for i in range(n_tasks):
        ade, fde = validate(model, datasets[i]) if (i<current_task) else (0.,0.)
        ades_all_classes.append(ade)
        fdes_all_classes.append(fde)

    average_ades = sum([ades_all_classes[task_id] for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes_all_classes[task_id] for task_id in range(current_task)]) / current_task

    for task_id in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(task_id+1)].append(ades_all_classes[task_id])
        metrics_dict["fde per task"]["task {}".format(task_id+1)].append(fdes_all_classes[task_id])

    metrics_dict["average_ade"].append(average_ades)
    metrics_dict["average_fde"].append(average_fdes)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen

    return metrics_dict



###---------------------------------------------------------------------------------------------------###

##-----------------------------##
##----PREDICTION EVALUATION----##
##-----------------------------##

def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade_ = utils.displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_ = utils.final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade_, fde_

def evaluate(loader, predictor):
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

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            pred_len = pred_traj_gt.size(0)

            for _ in range(20):
                pred_traj_fake_rel = predictor(obs_traj_rel)
                pred_traj_fake = utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)


            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer).item() / (total_traj * pred_len)
        fde = sum(fde_outer).item() / (total_traj)
        # a = sum(ade_outer)
        # b = total_traj * args.pred_len
        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)
        return ade, fde


def validate(model, dataset_name, batch_size=128, test_size=1024, verbose=True):
    '''
    Evaluate precision (ADE and FDE) of a predictor ([model]) on [dataset].
    '''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Loop over batches in [dataset]
    '''
    print("\nInitializing test dataset")
    test_path = utils.get_dset_path("test")
    data_type = ".txt"
    _, test_loader = data_loader(args, test_path, dataset_name, data_type)
    '''
    ade, fde = evaluate(dataset_name, model)

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    '''
    if verbose:
        print("\nDataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            dataset_name, args.pred_len, ade, fde
        ))
    '''
    return ade, fde

def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
              test_size=None, visdom=None, verbose=False, summary_graph=True):
    n_tasks = len(datasets)
    ades = []
    fdes = []
    for i in range(n_tasks):
        if i+1 <=current_task:
            ade, fde = validate(model, datasets[i])
            ades.append(ade)
            fdes.append(fde)
        else:
            ades.append(0)
            fdes.append(0)

    average_ades = sum([ades[task_id] for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes[task_id] for task_id in range(current_task)]) / current_task

    # Send results to visdom server
    names = ['task {}'.format(i+1) for i in range(n_tasks)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            ades, names=names, title="ADE ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="ADE precision"
        )
        visual_visdom.visualize_scalars(
            fdes, names=names, title="FDE ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="FDE precision"
        )
        if n_tasks > 1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_ades], names=["ADE"], title="Average ADE precision ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="ADE precision"
            )
            visual_visdom.visualize_scalars(
                [average_fdes], names=["FDE"], title="Average FDE precision ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="FDE precision"
            )






