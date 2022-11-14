import argparse
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import torch

from data.loader import data_loader
from main_model.encoder import Predictor
from helper.evaluate import evaluate
from helper import utils
from helper.utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)


torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="ETH", help="Directory containing logging file")
parser.add_argument("--dataset_name_train", default="ETH", type=str)
parser.add_argument("--dataset_name_test", default="ETH", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=8, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=200, type=int)
parser.add_argument("--val_epoch", default=2, type=int)
augmentation_choices = ["none", "rotation"]
parser.add_argument("--aug", type=str, default='none', choices=augmentation_choices, help="whether to rotation the data")

parser.add_argument("--noise_dim", default=(8,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

# lstm
parser.add_argument("--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size")
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument('--traj_lstm_output_size', default=32, type=int)
# gat
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after through GAT module")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
model_choices = ["lstm", "gat"]
parser.add_argument('--main_model', default='lstm', type=str, choices=model_choices, help="the main model of CL and BL")

parser.add_argument("--num_samples", default=20, type=int)
parser.add_argument("--dset_type", default="test", type=str)
parser.add_argument("--resume", default="model_best.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)",)


def get_generator(checkpoint):
    if args.main_model == "lstm":
        from main_model.encoder import Predictor
        model = Predictor(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size,
            traj_lstm_output_size=args.traj_lstm_output_size
        )
    if args.main_model == "gat":
        n_units = (
                [args.traj_lstm_hidden_size]
                + [int(x) for x in args.hidden_units.strip().split(",")]
                + [args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        from main_model.encoder_gat import Predictor
        model = Predictor(
            obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
            n_units=n_units, n_heads=n_heads, graph_network_out_dims=args.graph_network_out_dims,
            dropout=args.dropout, alpha=args.alpha, graph_lstm_hidden_size=args.graph_lstm_hidden_size
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model



def main(args):
    checkpoint_path = os.path.join(os.path.abspath(args.log_dir), "{}_{}_{}_best.pth.tar".format(args.main_model, args.dataset_name_train, args.aug))
    checkpoint = torch.load(checkpoint_path)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name_test, args.dset_type)

    _, loader = data_loader(args, path)
    ade, fde = evaluate(loader, generator)
    d = {'training dataset': args.dataset_name_train, 'testing dataset': args.dataset_name_test, 'Pred len': args.pred_len,
         'ADE': ade, 'FDE': fde}
    utils.save_dict(d, "batch_learning_{}_{}_{}_{}".format(args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
    utils.save_dict_txt(d, "batch_learning_{}_{}_{}_{}".format(args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
    print(
        "Train Dataset: {} | Test Dataset: {} | Pred Len: {} | ADE: {:.12f} | FDE: {:.12f}".format(
            args.dataset_name_train, args.dataset_name_test, args.pred_len, ade, fde
        )
    )



if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
