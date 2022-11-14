from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_dset(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    return dset
def data_loader(args, dset, batch_size, shuffle=False):
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return loader

