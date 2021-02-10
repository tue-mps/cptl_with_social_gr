#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
import utils
from torch import optim
import pandas as pd

from param_values import set_default_values
from vae_models import AutoEncoder
from encoder import Predictor
from replayer import Replayer
from train import train_cl
from data import data_loader

parser = argparse.ArgumentParser('./main.py', description='Run experiment.')
parser.add_argument('--get-stamp', action='store_true', help="print param-stamp & exit")
parser.add_argument('--seed', type=int, default=0, help="random seed (for each random-module used)")
parser.add_argument('--no-gups', action='store_false', dest='cuda', help="do not use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default")

parser.add_argument('--dataset_name', default="univ", type=str, help="the training dataset path")
parser.add_argument('--obs_len', default=12, type=int, help="the observed frame of trajectory")
parser.add_argument('--pred_len', default=8, type=int, help="the predicted frame of trajectory")
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--delim', default='\t')
parser.add_argument('--batch_size', default=64, type=int)

# experimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--tasks', type=int, help="number of tasks")

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary classfication loss")
loss_params.add_argument('--bce_distill', action='store_true', help="distilled loss on previous classes for new")

# model architecture parameters, about LSTM #todo
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('')

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size") #todo reference batch_size
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback', action='store_true', help="equip model with feedback connections")
replay_params.add_argument('--z-dim', type=int, default=100, help="size of latent representation")
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
replay_params.add_argument('--distill', action='store_true', help='use distillation for replay')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--agem', action='store_true', help="use gradient of replay as inequality constraint")

# generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help="size of latent representation")
genmodel_params.add_argument('--g-fc-lay', type=int, help="[fc_layers] in generator (default: same as lstm)")
genmodel_params.add_argument('--g-fc-uni', type=int, help="[fc_units] in generator (default: same as lstm)")

# hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="batches to train generator (default: same as lstm)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: same as lr)")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
eval_params.add_argument('--pdf', action='store_true', help="generator pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="samples for evaluating solver's precision")
eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="iters after which to plot samples")
eval_params.add_argument('--sample-n', type=int, default=64, help="images to show")  #todo



def run(args, verbose=False):

    # Set default arguments & check for incompatible options
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log =args.iters
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    #------------------------------------------------------------------------------------------------#

    #----------------#
    #------data------#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
    # todo
    # train_datasets = None
    # test_datasets = None
    train_path = utils.get_dset_path(args.dataset_name, "train")
    val_path = utils.get_dset_path(args.dataset_name, "val")

    print("\nInitializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    print("\nInitializing val dataset")
    _, val_loader = data_loader(args, val_path)



    #--------------------------------------------------------------------------------------------------#

    #--------------------#
    #----Model (LSTM)----#
    #--------------------#

    # Define main model (i.e., lstm, if requested with feedback connections) #todo
    if args.feedback:
        model = AutoEncoder().to(device)
        model.lamda_pl = 1.
    else:
        model = Predictor().to(device)

    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.required_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type == "sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))


    #-----------------------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----CL-STRATEGY: REPLAY----#
    #---------------------------#  #todo

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder().to(device)
        # -set optimizer(s)
        generator.optim_list = [{'params': filter(lambda p: p.required_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None


    #------------------------------------------------------------------------------------------------------------------#

    #--------------------#
    #------REPORTING-----#
    #--------------------#  #todo

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    #todo
    param_stamp = None

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        pass
        # -generator

    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.pdf or args.metrics:
        pass
    else:
        pass

    # -Prepare for plotting in visdom
    if args.visdom:
        pass
    else:
        pass


    #----------------------------------------------------------------------------------------------------------------#

    #-----------------#
    #----CALLBACKS----#
    #-----------------#  #todo

    # Callbacks for reporting and visualizing accuracy
    generator_loss_cbs = [] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    eval_cbs = [] if (not args.use_exemplars) else [None]

    # Callbacks for calculating statists required for metrics
    metric_cbs = []


    #-----------------------------------------------------------------------------------------------------------------#

    #----------------#
    #----TRAINING----#
    #----------------#

    if verbose:
        print("\nTraining...")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl()
    # Get total training-time in seconds, and write to file
    if args.time:
        training_time = time.time() - start
        time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
        time_file.write('{}\n'.format(training_time))
        time_file.close()


    #------------------------------------------------------------------------------------------------------------------#

    #------------------#
    #----EVALUATION----#
    #------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [None for i in range(args.tasks)]
    average_prec = sum(precs) / args.tasks
    # -print on screen
    if verbose:
        print("\n Precision on test-set{}:".format("(softmax classification)" if args.use_exemplars else ""))
        for i in range(args.tasks):
            print("- Task {}: {:.4f}".format(i+1, precs[i]))
        print("=> Average precision over all {} tasks: {:.4f}\n".format(args.tasks, average_prec))

    if verbose and args.time:
        print("=> Total training time = {:.1f} seconds\n".format(training_time))



    #------------------------------------------------------------------------------------------------------------------#

    #-------------------#
    #------OUTPUT-------#
    #-------------------# #todo

    # Average precision on full test set

    # -metrics-dict


    #------------------------------------------------------------------------------------------------------------------#

    #-----------------#
    #----PLOTTING-----#
    #-----------------#

    # If requested, generate pdf
    if args.pdf:
        pass





if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)