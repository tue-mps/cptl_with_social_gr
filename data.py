from torch.utils.data import DataLoader
from trajectories import TrajectoryDataset, seq_collate

def data_loader(args, path, data_name, data_type):
    dset = TrajectoryDataset(
        path,
        data_name,
        data_type,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True
    )
    return dset, loader