from torch.utils.data import DataLoader

from data.trajectories_interaction import TrajectoryDataset, seq_collate


def data_dset_interaction(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    return dset
def data_loader_interaction(args, dset, batch_size):
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return loader

