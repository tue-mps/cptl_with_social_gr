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

def memory_buff(args):
    # eth
    train_path_eth = utils.get_dset_path("ETH", 'train')
    train_dset_eth = data_dset(args, train_path_eth)
    num_memory_eth = int(0.1 * len(train_dset_eth))
    dataset_eth = data_loader(args, train_dset_eth, num_memory_eth)
    batch_eth = []
    for batch_index, batch in enumerate(dataset_eth):
        batch_eth = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch_eth

        out = [
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            seq_start_end
        ]
        break

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
        # num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True
    )

    # print("Finished")
    return loader, batch_eth