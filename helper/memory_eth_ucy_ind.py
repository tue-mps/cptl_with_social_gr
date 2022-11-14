import random

import numpy as np
import torch

from data.loader import data_loader, data_dset
from helper import utils
import argparse
from data.trajectories_memory import TrajectoryDataset, seq_collate
from torch.utils.data import DataLoader

# parser = argparse.ArgumentParser('./main.py', description='Run experiment.')
# parser.add_argument('--obs_len', default=8, type=int, help="the observed frame of trajectory")
# parser.add_argument('--pred_len', default=12, type=int, help="the predicted frame of trajectory")
# parser.add_argument('--skip', default=1, type=int)
# parser.add_argument('--delim', default='\t')
# parser.add_argument('--loader_num_workers', default=8, type=int)
# args = parser.parse_args()

def seq_collate_(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list) = zip(*data)
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    obs_traj_ = torch.cat(obs_seq_list, dim=0).permute(1,0,2)
    pred_traj_ = torch.cat(pred_seq_list, dim=0).permute(1,0,2)
    obs_traj_rel_ = torch.cat(obs_seq_rel_list, dim=0).permute(1,0,2)
    pred_traj_rel_ = torch.cat(pred_seq_rel_list, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj_,
        pred_traj_,
        obs_traj_rel_,
        pred_traj_rel_,
        seq_start_end
    ]

    return tuple(out)


def memory_buff(args, batch_eth, batch_ucy):
    # # eth
    # train_path_eth = utils.get_dset_path("ETH", 'train')
    # train_dset_eth = data_dset(args, train_path_eth)
    # num_memory_eth = int(0.1 * len(train_dset_eth))
    # dataset_eth = data_loader(args, train_dset_eth, num_memory_eth)
    # batch_eth = []
    # for batch_index, batch in enumerate(dataset_eth):
    #     batch_eth = [tensor.cuda() for tensor in batch]
    #     break

    # print("one eth batch")

    # # ucy
    # train_path_ucy = utils.get_dset_path("UCY", 'train')
    # train_dset_ucy = data_dset(args, train_path_ucy)
    # num_memory_ucy = int(0.1 * len(train_dset_ucy))
    # dataset_ucy = data_loader(args, train_dset_ucy, num_memory_ucy)
    # batch_ucy = []
    # for batch_index, batch in enumerate(dataset_ucy):
    #     batch_ucy = [tensor.cuda() for tensor in batch]
    #     break

    # print("one ucy batch")


    # ind
    train_path_ind = utils.get_dset_path("inD", 'train')
    train_dset_ind = data_dset(args, train_path_ind)
    num_memory_ind = int(0.1 * len(train_dset_ind))
    dataset_ind = data_loader(args, train_dset_ind, num_memory_ind)
    batch_ind = []
    for batch_index, batch in enumerate(dataset_ind):
        batch_ind = [tensor.cuda() for tensor in batch]
        break

    # print("one ind batch")





    obs_traj = torch.cat((batch_ucy[0], batch_ind[0], batch_eth[0]), dim=1)
    pred_traj = torch.cat((batch_ucy[1], batch_ind[1], batch_eth[1]), dim=1)
    obs_traj_rel = torch.cat((batch_ucy[2], batch_ind[2], batch_eth[2]), dim=1)
    pred_traj_rel = torch.cat((batch_ucy[3], batch_ind[3], batch_eth[3]), dim=1)
    seq_start_end_eth = batch_eth[6]
    _, end_eth = seq_start_end_eth[-1]
    seq_start_end_ucy = batch_ucy[6]
    _, end_ucy = seq_start_end_ucy[-1]
    seq_start_end_ind = batch_ind[6]
    _, end_ind = seq_start_end_ind[-1]

    # seq_start_end_ucy = seq_start_end_ucy + end_eth
    # seq_start_end_ind = seq_start_end_ind + end_eth + end_ucy

    seq_start_end_ind = seq_start_end_ind + end_ucy
    seq_start_end_eth = seq_start_end_eth + end_ucy + end_ind

    seq_start_end = torch.cat((seq_start_end_ucy, seq_start_end_ind, seq_start_end_eth), dim=0)

    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        seq_start_end
    ]

    dset = TrajectoryDataset(
        out[0].detach().cpu(),
        out[1].detach().cpu(),
        out[2].detach().cpu(),
        out[3].detach().cpu(),
        out[4].detach().cpu(),
    )
    # print("create dset")
    loader = DataLoader(
        dset,
        batch_size=args.replay_batch_size,
        shuffle=True,
        # num_workers=4,
        collate_fn=seq_collate,
        pin_memory=True
    )




    # index = random.sample(range(0, seq_start_end.size(0)), 64)
    # index.sort()
    # out = []
    # for i in range(len(index)):
    #     out_ = [
    #         obs_traj[index[i]] if len(obs_traj[index[i]].size())==3 else obs_traj[index[i]].unsqueeze(dim=0),
    #         pred_traj[index[i]] if len(pred_traj[index[i]].size())==3 else pred_traj[index[i]].unsqueeze(dim=0),
    #         obs_traj_rel[index[i]] if len(obs_traj_rel[index[i]].size())==3 else obs_traj_rel[index[i]].unsqueeze(dim=0),
    #         pred_traj_rel[index[i]] if len(pred_traj_rel[index[i]].size())==3 else pred_traj_rel[index[i]].unsqueeze(dim=0)
    #     ]
    #     out.append(out_)
    #
    # memory_seq = seq_collate_(out)

    # print("Finished")
    return loader, batch_ind