import logging
import os
import math
from IPython import embed
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        seq_start_end,
    ]

    return tuple(out)     # tuple(out)




class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        seq_start_end
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()



        # Convert numpy -> Torch Tensor
        self.obs_traj = obs_traj.permute(1,2,0)
        self.pred_traj = pred_traj.permute(1,2,0)
        self.obs_traj_rel = obs_traj_rel.permute(1,2,0)
        self.pred_traj_rel = pred_traj_rel.permute(1,2,0)
        self.seq_start_end = seq_start_end
        self.num_seq = obs_traj.size(0)


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
        ]
        return out
