#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
import utils
import evaluate
import visual_plt
import callbacks as cb
from torch import optim
import pandas as pd

from param_values import set_default_values
from vae_models import AutoEncoder
from encoder import Predictor
from replayer import Replayer
from train import train_cl
from data import data_set,data_loader
from param_stamp import get_param_stamp

parser = argparse.ArgumentParser('./main.py', description='Run experiment.')
parser.add_argument('--get-stamp', action='store_true', help="print param-stamp & exit")
parser.add_argument('--seed', type=int, default=0, help="random seed (for each random-module used)")
parser.add_argument('--no-gups', action='store_false', dest='cuda', help="do not use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default")

parser.add_argument('--dataset_name', default="univ", type=str, help="the training dataset path")
parser.add_argument('--obs_len', default=8, type=int, help="the observed frame of trajectory")
parser.add_argument('--pred_len', default=12, type=int, help="the predicted frame of trajectory")
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--delim', default='\t')
# parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--loader_num_workers', default=8, type=int)

# experimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--tasks', type=int, help="number of tasks")

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary classfication loss")
loss_params.add_argument('--bce_distill', action='store_true', help="distilled loss on previous classes for new")

# model architecture parameters, about LSTM #
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--traj_lstm_input_size', default=2, type=int)
model_params.add_argument('--traj_lstm_hidden_size', default=32, type=int)
model_params.add_argument('--traj_lstm_output_size', default=32, type=int)


# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, default=5, help="batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch_size', type=int, default=64, help="batch-size") #
train_params.add_argument('--replay_batch_size', default=32, type=int, help="replay batch size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback', action='store_true', help="equip model with feedback connections")
replay_params.add_argument('--z_dim', type=int, default=100, help="size of latent representation")
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
replay_params.add_argument('--replay', type=str, default='generative', choices=replay_choices)
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

# data storage ('exemplars') parameters
store_params = parser.add_argument_group('Data Storage Parameters')
store_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")

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
eval_params.add_argument('--num_samples', type=int, default=20, help="sample trajectories when evaluation model")



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
    #
    # train_datasets = None
    # test_datasets = None
    # train_order = ['students003','students001','crowds_zara02','crowds_zara01','biwi_hotel','biwi_eth']
    train_order = ['uni_examples','students003','students001','crowds_zara03','biwi_hotel','crowds_zara02','crowds_zara01']
    val_order = ['biwi_eth','biwi_eth','biwi_eth','biwi_eth','biwi_eth','biwi_eth','biwi_eth']
    val_name = "biwi_eth"
    # train_order = ['biwi_hotel','uni_examples','students003','students001','crowds_zara03','crowds_zara02','crowds_zara01','biwi_eth']
    # train_order = ['uni_examples','students003','students001','crowds_zara03','crowds_zara02','crowds_zara01','biwi_eth']
    tasks = len(train_order)
    # test_dataset = "biwi_eth"
    train_datasets = []
    val_datasets = []
    print("\nInitializing train dataset")
    print("\nInitializing val dataset")

    for i, dataset_name in enumerate(train_order):
        # load train/val dataset path
        train_path = utils.get_dset_path("train")
        # val_path = utils.get_dset_path("val")
        # load train dataset
        data_type = ".txt"
        train_dset = data_set(args, train_path, dataset_name, data_type)
        # load val dataset
        # data_type = "_val.txt"
        # _, val_loader = data_loader(args, val_path, dataset_name, data_type)
        train_datasets.append(train_dset)
        # val_datasets.append(val_loader)

    for i, dataset_name in enumerate(val_order):
        # load train/val dataset path
        # train_path = utils.get_dset_path("train")
        val_path = utils.get_dset_path("val")
        # load train dataset
        # data_type = "_train.txt"
        # train_dset, train_loader = data_loader(args, train_path, dataset_name, data_type)
        # load val dataset
        data_type = ".txt"
        val_dset = data_set(args, val_path, dataset_name, data_type)
        val_loader = data_loader(args, val_dset, args.batch_size)
        # train_datasets.append(train_loader)
        val_datasets.append(val_loader)



    #--------------------------------------------------------------------------------------------------#

    #--------------------#
    #----Model (LSTM)----#
    #--------------------#

    # Define main model (i.e., lstm, if requested with feedback connections) #todo
    if args.feedback:
        model = AutoEncoder(obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                            traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
                            z_dim=args.z_dim).to(device)
        model.lamda_pl = 1.
    else:
        model = Predictor(obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                          traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size).to(device)

    # Define optimizer (only include parameters that "requires_grad")
    # model.optim_list = [{'params': filter(lambda p: p.required_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        # model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        model.optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
    elif model.optim_type == "sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))


    #-----------------------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----CL-STRATEGY: REPLAY----#
    #---------------------------#  #

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder(obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                                traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
                                z_dim=args.z_dim).to(device)
        # -set optimizer(s)
        # generator.optim_list = [{'params': filter(lambda p: p.required_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            # generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
            generator.optimizer = optim.Adam(generator.parameters(), betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None


    #------------------------------------------------------------------------------------------------------------------#

    #--------------------#
    #------REPORTING-----#
    #--------------------#  #

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, model.name, verbose=verbose, replay=True if (not args.replay=="none") else False,
        replay_model_name=generator.name if (args.replay=="generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.pdf or args.metrics:
        metric_dict = evaluate.initiate_metrics_dict(n_tasks=tasks)
        metric_dict = evaluate.intial_accuracy(model, val_datasets, metric_dict)
    else:
        metric_dict = None

    # -Prepare for plotting in visdom
    # -visdom-settings
    if args.visdom:
        env_name = "{exp}-{tasks}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{val_name}".format(exp='Pedestrian', tasks=tasks, iters=args.iters, z_dim=args.z_dim, batch_size=args.batch_size, replay_batch_size=args.replay_batch_size, lr=args.lr, val_name=val_name)
        graph_name = "{fb}{replay}".format(
            fb="1M-" if args.feedback else "",
            replay="{}{}{}".format(args.replay, "D" if args.distill else "", "-aGEM" if args.agem else ""),
        )
        visdom = {'env': env_name, 'graph': graph_name}
    else:
        visdom = None


    #----------------------------------------------------------------------------------------------------------------#

    #-----------------#
    #----CALLBACKS----#
    #-----------------#  #

    # Callbacks for reporting and visualizing accuracy
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, model=model if args.feedback else generator, tasks=tasks,
                        iters_per_task=args.iters if args.feedback else args.g_iters,
                        replay=False if args.replay=="none" else True)
    ] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=tasks,
                           iters_per_task=args.iters, replay=False if args.replay=="none" else True)
    ] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    eval_cbs = [
        cb._eval_cb(log=args.prec_log, test_datasets=val_datasets, visdom=visdom,
                    iters_per_task=args.iters)
    ] if (not args.use_exemplars) else [None]

    # Callbacks for calculating statists required for metrics
    metric_cbs = [
        cb._metric_cb(log=args.iters, test_datasets=val_datasets,
                      iters_per_task=args.iters, metrics_dict=metric_dict)
    ]



    #-----------------------------------------------------------------------------------------------------------------#

    #----------------#
    #----TRAINING----#
    #----------------#

    if verbose:
        print("\nTraining...")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(args, model, train_datasets, replay_model=args.replay, iters=args.iters, batch_size=args.batch_size,
             generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
             sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
             metric_cbs=metric_cbs)
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
    ades = []
    fdes = []
    for i in range(tasks):
        ade, fde = evaluate.validate(model, val_datasets[i])
        ades.append(ade)
        fdes.append(fde)
    average_ades = sum(ades) / tasks
    average_fdes = sum(fdes) / tasks
    # -print on screen
    if verbose:
        print("\n Precision on test-set")
        for i in range(tasks):
            print(" - Task {}: ADE {:.4f} FDE {:.4f}".format(i+1, ades[i], fdes[i]))
        print("==> Average precision over all {} tasks: ADE {:.4f} FDE {:.4f}".format(tasks, average_ades, average_fdes))





    # -print on screen
    # if verbose:
    #     print("\n Precision on test-set{}:".format("(softmax classification)" if args.use_exemplars else ""))
    #     for i in range(args.tasks):
    #         print("- Task {}: {:.4f}".format(i+1, precs[i]))
    #     print("=> Average precision over all {} tasks: {:.4f}\n".format(args.tasks, average_prec))

    if verbose and args.time:
        print("=> Total training time = {:.1f} seconds\n".format(training_time))



    #------------------------------------------------------------------------------------------------------------------#

    #-------------------#
    #------OUTPUT-------#
    #-------------------# #

    # Average precision on full test set
    output_file = open("{}/prec-{replay}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{val_name}.txt".format(args.r_dir, replay=args.replay, iters=args.iters, z_dim=args.z_dim, batch_size=args.batch_size, replay_batch_size=args.replay_batch_size, lr=args.lr, val_name=val_name), "w")
    output_file.write('Training:{order}\nADEs:{ades}\nADE:{ade}\nFDEs:{fdes}\nFDE:{fde}'.format(order=train_order,ades=ades, ade=average_ades, fdes=fdes, fde=average_fdes))
    output_file.close()

    # -metrics-dict


    #------------------------------------------------------------------------------------------------------------------#

    #-----------------#
    #----PLOTTING-----#
    #-----------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}-{}.pdf".format(args.p_dir, param_stamp, val_name)
        pp = visual_plt.open_pdf(plot_name)

        # -show metrics reflecting progression during training
        figure_list = []  # -> create list to store all figures to be plotted

        # -generate all figures (and store them in [figure_list])
        key_ade = "ade per task"
        plot_ade_list = []
        for i in range(tasks):
            plot_ade_list.append(metric_dict[key_ade]["task {}".format(i+1)])
        figure = visual_plt.plot_lines(
            plot_ade_list, x_axes=metric_dict["x_task"],
            line_names=["task {}".format(i+1) for i in range(tasks)],
            title="ADE for each tasks"
        )
        figure_list.append(figure)

        key_fde = "fde per task"
        plot_fde_list = []
        for i in range(tasks):
            plot_fde_list.append(metric_dict[key_fde]["task {}".format(i+1)])
        figure = visual_plt.plot_lines(
            plot_fde_list, x_axes=metric_dict["x_task"],
            line_names=["task {}".format(i+1) for i in range(tasks)],
            title="FDE for each tasks"
        )
        figure_list.append(figure)

        # calculate average ade/fde
        figure = visual_plt.plot_lines(
            [metric_dict["average_ade"]], x_axes=metric_dict["x_task"],
            line_names=["average ade all tasks so far"],
            title="Average ADE"
        )
        figure_list.append(figure)

        figure = visual_plt.plot_lines(
            [metric_dict["average_fde"]], x_axes=metric_dict["x_task"],
            line_names=["average fde all tasks so far"],
            title="Average FDE"
        )
        figure_list.append(figure)

        # -add figures to pdf (and close this pdf)
        for figure in figure_list:
            pp.savefig(figure)

        # output
        output_file = open(
            "{}/ADE-FDE-{replay}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{val_name}.txt".format(args.r_dir, replay=args.replay,
                                                                                       iters=args.iters,
                                                                                       z_dim=args.z_dim,
                                                                                       batch_size=args.batch_size,
                                                                                       replay_batch_size=args.replay_batch_size,
                                                                                       lr=args.lr, val_name=val_name),
            "w")
        output_file.write('ADEs:{ades}\nAverage_ADE:{ade}\nFDEs:{fdes}\nAverage_FDE:{fde}'.format(ades=plot_ade_list,ade=metric_dict["average_ade"],fdes=plot_fde_list,fde=metric_dict["average_fde"]))
        output_file.close()


        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))





if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)