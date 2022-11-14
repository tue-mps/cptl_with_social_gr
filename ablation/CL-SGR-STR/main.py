#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter

from train import train_cl, train
from helper.replayer import Replayer
from helper import evaluate
from helper import visual_plt
from helper import callbacks as cb
from helper import utils
from helper.param_values import set_default_values
from helper.param_stamp import get_param_stamp
from helper.continual_learner import ContinualLearner

parser = argparse.ArgumentParser('./main.py', description='Run experiment.')
parser.add_argument('--get-stamp', action='store_true', help="print param-stamp & exit")
parser.add_argument('--seed', type=int, default=72, help="random seed (for each random-module used)")
parser.add_argument('--no-gups', action='store_false', dest='cuda', help="do not use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default")
dataset_choices = ['pedestrian', 'vehicle', 'interaction']
parser.add_argument('--dataset_order', type=str, default='vehicle', choices=dataset_choices)
method_choices = ['batch_learning', 'continual_learning']
parser.add_argument('--method', type=str, default='batch_learning', choices=method_choices)

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, default=200, help="batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
train_params.add_argument('--batch_size', type=int, default=32, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')
train_params.add_argument('--obs_len', default=8, type=int, help="the observed frame of trajectory")
train_params.add_argument('--pred_len', default=12, type=int, help="the predicted frame of trajectory")
train_params.add_argument('--skip', default=1, type=int)
train_params.add_argument('--delim', default='\t')
train_params.add_argument('--loader_num_workers', default=8, type=int)
train_params.add_argument("--gpu_index", default=0, type=int)
augmentation_choices = ["none", "rotation"]
train_params.add_argument("--aug", type=str, default='none', choices=augmentation_choices, help="whether to rotation the data")
train_params.add_argument('--val_epoch', default=150, type=int, help="epoch start to validation")

#########################################################################################################################
# main model architecture parameters #
# lstm
model_params = parser.add_argument_group('Main Model Parameters')
model_params.add_argument('--traj_lstm_input_size', default=2, type=int)
model_params.add_argument('--traj_lstm_hidden_size', default=32, type=int)
model_params.add_argument('--traj_lstm_output_size', default=32, type=int)
# gat
model_params.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
model_params.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
model_params.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after through GAT module")
model_params.add_argument("--graph_lstm_hidden_size", default=32, type=int)
model_params.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
model_params.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
model_choices = ["lstm", "gat"]
model_params.add_argument('--main_model', default='lstm', type=str, choices=model_choices, help="the main model of CL and BL")

# train-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="batches to train generator (default: same as lstm)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: same as lr)")
gen_params.add_argument('--replay_batch_size', type=int, default=32, help="replay batch size, it is same with batch size")

# "memory replay" parameters
replay_params = parser.add_argument_group('Generative Replay Parameters')
replay_params.add_argument('--z_dim', type=int, default=100, help="size of latent representation")
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
replay_params.add_argument('--x_dim', default=2, type=int)
replay_params.add_argument('--h_dim', default=64, type=int)
replay_params.add_argument('--n_layers', default=1, type=int)

replay_model_choices = ['lstm', 'vrnn']
replay_params.add_argument('--replay_model', default='lstm', type=str, choices=replay_model_choices, help="the generative replay model of CL")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")



# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
eval_params.add_argument('--pdf', action='store_true', help="generator pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--val', action='store_true', help="use validation data")
class_choices = ['current', 'all', 'replay']
eval_params.add_argument('--val_class', default='current', type=str, choices=class_choices, help='whether use current or previous task validation data')
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=50, metavar="N", help="iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=50, metavar="N", help="iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="samples for evaluating solver's precision")
eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="iters after which to plot samples")
eval_params.add_argument('--num_samples', type=int, default=20, help="sample trajectories when evaluation model")

##########################################################################################################################
# batch learning
batch_params = parser.add_argument_group('Batch learning Parameters')
batch_params.add_argument("--log_dir", default="ETH", help="Directory containing logging file")
batch_params.add_argument("--dataset_name", default="ETH", type=str)
batch_params.add_argument("--start_epoch", default=1, type=int, metavar="N", help="manual epoch number (useful on restarts)")
batch_params.add_argument("--print_every", default=10, type=int)
batch_params.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
batch_params.add_argument("--checkpoint_log", default=50, type=int, help="iters after which to save checkpoint")




def run(args, verbose=False):

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(device)
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    ###############################################################################
    ##################
    ##Batch learning##
    ##################
    if args.method == "batch_learning":
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        utils.set_logger(os.path.join(os.path.abspath(args.log_dir), "train.log"))
        checkpoint_dir = args.log_dir + "/checkpoint"
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)
        train_path = utils.get_dset_path(args.dataset_name, "train")
        val_path = utils.get_dset_path(args.dataset_name, "val")

        # loader data
        logging.info("Initializing train dataset")
        if args.aug == "none":
            from data.loader import data_loader,data_dset
        else:
            from data.loader_rotation import data_loader
        train_dset = data_dset(args, train_path)
        train_loader = data_loader(args, train_dset, args.batch_size)
        val_dset = data_dset(args, val_path)
        val_loader = data_loader(args, val_dset, args.batch_size)
        writer = SummaryWriter()
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
            from main_model.encoder_gat import Predictor
            model = Predictor(
                obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
                n_units=n_units, n_heads=n_heads, graph_network_out_dims=args.graph_network_out_dims,
                dropout=args.dropout, alpha=args.alpha, graph_lstm_hidden_size=args.graph_lstm_hidden_size
            )
        model.cuda()
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
        best_ade = 200
        if args.resume:
            if os.path.isfile(args.resume):
                logging.info("Restoring from checkpoint {}".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                logging.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

        for epoch in range(args.start_epoch, args.iters + 1):
            train(args, model, train_loader, optimizer, epoch, writer)
            if epoch >= args.val_epoch:
                ade = utils.validate(args, model, val_loader, epoch, writer=writer)
                is_best = ade < best_ade
                best_ade = min(ade, best_ade)
                # if epoch % args.checkpoint_log == 0:
                save_checkpoint_path = args.log_dir + "/checkpoint"
                utils.save_checkpoint(
                    args,
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    save_checkpoint_path + f"/{model.name}_checkpoint{epoch}_{args.aug}.pth.tar",
                    model_name=model.name,
                )

        writer.close()


    ###############################################################################
    ######################
    ##Continual learning##
    ######################

    if args.method == "continual_learning":
        # Set default arguments & check for incompatible options
        args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
        args.g_iters = args.iters if args.g_iters is None else args.g_iters
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

        #------------------------------------------------------------------------------------------------#
        #----------------#
        #------data------#
        #----------------#

        # Prepare data for chosen experiment
        if verbose:
            print("\nPreparing the data...")
        #
        if args.dataset_order == 'pedestrian':
            train_order = ['ETH', 'inD', 'INTERACTION', 'UCY']
            val_order = ['ETH', 'inD', 'INTERACTION', 'UCY']
            test_order = ['ETH', 'inD', 'INTERACTION', 'UCY']
        if args.dataset_order == 'vehicle':
            train_order = ['highD', 'inD', 'rounD']
            val_order = ['highD', 'inD', 'rounD']
            test_order = ['highD', 'inD', 'rounD']
        if args.dataset_order == 'interaction':
            train_order = ['MT', 'SR', 'LN', 'OF']
            val_order = ['MT', 'SR', 'LN', 'OF']
            test_order = ['MT', 'SR', 'LN', 'OF']


        tasks = len(train_order)
        train_datasets = []
        val_datasets = []
        val_dataset = []
        test_datasets = []

        if args.aug == "none":
            from data.loader import data_loader, data_dset
        else:
            from data.loader_rotation import data_loader

        print("\nInitializing train dataset")
        for i, dataset_name in enumerate(train_order):
            # load train/val dataset path
            train_path = utils.get_dset_path(dataset_name, 'train')
            # load train dataset
            train_dset = data_dset(args, train_path)
            # train_loader = data_loader(args, train_dset, args.batch_size)
            print("dataset: {} | train trajectories: {}".format(dataset_name, (train_dset.obs_traj.shape[0])))
            train_datasets.append(train_dset)

        print("\nInitializing val dataset")
        for i, dataset_name in enumerate(val_order):
            # load val dataset path
            val_path = utils.get_dset_path(dataset_name, "val")
            # load val dataset
            val_dset = data_dset(args, val_path)
            val_loader = data_loader(args, val_dset, args.batch_size)
            print("dataset: {} | val trajectories: {}".format(dataset_name, (val_dset.obs_traj.shape[0])))
            val_datasets.append(val_dset)
            val_dataset.append(val_loader)

        print("\nInitializing test dataset")
        for i, dataset_name in enumerate(test_order):
            # load test dataset path
            test_path = utils.get_dset_path(dataset_name, "test")
            # load test dataset
            test_dset = data_dset(args, test_path)
            test_loader = data_loader(args, test_dset, args.batch_size)
            print("dataset: {} | test trajectories: {}".format(dataset_name, (test_dset.obs_traj.shape[0])))
            test_datasets.append(test_loader)

        #--------------------------------------------------------------------------------------------------#
        #--------------------#
        #----Model (LSTM)----#
        #--------------------#

        # Define main model (i.e., lstm, if requested with feedback connections)
        if args.main_model == "lstm":
            from main_model.encoder import Predictor
            model = Predictor(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                traj_lstm_output_size=args.traj_lstm_output_size
            ).to(device)
        if args.main_model == "gat":
            from main_model.encoder_gat import Predictor
            model = Predictor(
                obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
                n_units=n_units, n_heads=n_heads, graph_network_out_dims=args.graph_network_out_dims,
                dropout=args.dropout, alpha=args.alpha, graph_lstm_hidden_size=args.graph_lstm_hidden_size
            ).to(device)

        # Define optimizer (only include parameters that "requires_grad")
        model.optim_type = args.optimizer
        if model.optim_type in ("adam", "adam_reset"):
            model.optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
        elif model.optim_type == "sgd":
            model.optimizer = optim.SGD(model.optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))
        best_ade = 200



        #-----------------------------------------------------------------------------------------------------------------#

        # Synpatic Intelligence (SI)
        if isinstance(model, ContinualLearner):
            model.si_c = args.si_c if args.si else 0
            if args.si:
                model.epsilon = args.epsilon

        #---------------------------#
        #----CL-STRATEGY: REPLAY----#
        #---------------------------#
        # If needed, specify separate model for the generator
        train_gen = True if (args.replay=="generative") else False
        if train_gen:
            fake_generator = None
            # -specify architecture
            # lstm
            if args.replay_model == 'lstm':
                from generative_model.vae_models import AutoEncoder
                generator = AutoEncoder(obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
                                        traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
                                        z_dim=args.z_dim).to(device)
            # vrnn
            if args.replay_model == 'vrnn':
                from generative_model.vae_models_VRNN import AutoEncoder
                generator = AutoEncoder(x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=args.n_layers, sample_k=args.num_samples, pred_len=args.obs_len).to(device)
            # -set optimizer(s)
            generator.optim_type = args.optimizer
            if generator.optim_type in ("adam", "adam_reset"):
                generator.optimizer = optim.Adam(generator.parameters(), betas=(0.9, 0.999))
            elif generator.optim_type == "sgd":
                generator.optimizer = optim.SGD(generator.optim_list)
        else:
            generator = None
            # lstm
            if args.replay_model == 'lstm':
                from generative_model.vae_models import AutoEncoder
                fake_generator = AutoEncoder(obs_len=args.obs_len, pred_len=args.pred_len,
                                             traj_lstm_input_size=args.traj_lstm_input_size,
                                             traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                                             traj_lstm_output_size=args.traj_lstm_output_size,
                                             z_dim=args.z_dim).to(device)
            # vrnn
            if args.replay_model == 'vrnn':
                from generative_model.vae_models_VRNN import AutoEncoder
                fake_generator = AutoEncoder(x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
                                             n_layers=args.n_layers, sample_k=args.num_samples,
                                             pred_len=args.obs_len).to(device)
            # -set optimizer(s)
            fake_generator.optim_type = args.optimizer
            if fake_generator.optim_type in ("adam", "adam_reset"):
                fake_generator.optimizer = optim.Adam(fake_generator.parameters(), betas=(0.9, 0.999))
            elif fake_generator.optim_type == "sgd":
                fake_generator.optimizer = optim.SGD(fake_generator.optim_list)

        #------------------------------------------------------------------------------------------------------------------#
        #--------------------#
        #------REPORTING-----#
        #--------------------#

        # Get parameter-stamp (and print on screen)
        if verbose:
            print("\nParameter-stamp...")
        param_stamp = get_param_stamp(
            args, model.name, verbose=verbose, replay=True if (not args.replay=="none") else False,
            replay_model_name=generator.name if (args.replay=="generative") else None,
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
            metric_dict = evaluate.intial_accuracy(model, test_datasets, metric_dict)
        else:
            metric_dict = None

        # -Prepare for plotting in visdom
        # -visdom-settings
        if args.visdom:
            env_name = "epoch-lstm-GR-lstm-replay-{exp}-{tasks}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{seed}-{val}-{val_class}-si{si}-{si_c}".format(exp=args.dataset_order, tasks=tasks, iters=args.iters, z_dim=args.z_dim, batch_size=args.batch_size, replay_batch_size=args.replay_batch_size, lr=args.lr, seed=args.seed, val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c)
            graph_name = "{fb}_{replay}".format(
                fb="{}".format('SI' if args.si else ''),
                replay="{}".format(args.replay),
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
            cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, model=generator, tasks=tasks,
                            iters_per_task=args.g_iters,
                            replay=False if args.replay=="none" else True)
        ] if (train_gen) else [None]
        fake_generator_loss_cbs = [
            cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, model=fake_generator, tasks=tasks,
                            iters_per_task=args.g_iters,
                            replay=False if args.replay=="none" else True)
        ] if (train_gen) is False else [None]
        solver_loss_cbs = [
            cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=tasks,
                               iters_per_task=args.iters, replay=False if args.replay=="none" else True)
        ]
        solver_val_loss_cbs = [
            cb._solver_val_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=tasks,
                               iters_per_task=args.iters, replay=False if args.replay=="none" else True)
        ]

        # Callbacks for evaluating and plotting generated / reconstructed samples
        sample_cbs = [] if (train_gen) else [None]

        # Callbacks for reporting and visualizing accuracy
        eval_cbs = [
            cb._eval_cb(log=args.prec_log, test_datasets=val_dataset, visdom=visdom,
                        iters_per_task=args.iters)
        ]

        # Callbacks for calculating statists required for metrics
        metric_cbs = [
            cb._metric_cb(log=args.iters, test_datasets=test_datasets,
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
        train_cl(args, best_ade, model, train_datasets, val_datasets, replay_model=args.replay, iters=args.iters, batch_size=args.batch_size,
                 generator=generator, fake_generator=fake_generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs, fake_gen_loss_cbs=fake_generator_loss_cbs,
                 sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs, val_loss_cbs=solver_val_loss_cbs,
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
            ade, fde = evaluate.validate(model, test_datasets[i])
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

        if verbose and args.time:
            print("=> Total training time = {:.1f} seconds\n".format(training_time))

        #------------------------------------------------------------------------------------------------------------------#
        #-------------------#
        #------OUTPUT-------#
        #-------------------# #

        # Average precision on full test set
        output_file = open("{}/prec-{replay}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{aug}-{main_model}-{train_order}-{seed}-{val}-{val_class}-si{si}-{si_c}.txt".format(args.r_dir, replay=args.replay, iters=args.iters, z_dim=args.z_dim, batch_size=args.batch_size, replay_batch_size=args.replay_batch_size, lr=args.lr, aug=args.aug, main_model=model.name, train_order=args.dataset_order, seed=args.seed, val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c), "w")
        output_file.write('Training:{order}\nADEs:{ades}\nADE:{ade}\nFDEs:{fdes}\nFDE:{fde}'.format(order=train_order,ades=ades, ade=average_ades, fdes=fdes, fde=average_fdes))
        output_file.close()

        #------------------------------------------------------------------------------------------------------------------#

        #-----------------#
        #----PLOTTING-----#
        #-----------------#

        # If requested, generate pdf
        if args.pdf:
            # -open pdf
            plot_name = "{}/{}-{}-{}-{}-{}-{}-{}-{}.pdf".format(args.p_dir, args.replay, args.iters, args.aug, args.seed, args.val, args.val_class, args.si, args.si_c)
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
                "{}/ADE-FDE-{replay}-{iters}-{z_dim}-{batch_size}-{replay_batch_size}-{lr}-{aug}-{main_model}-{train_order}-{seed}_{val}_{val_class}_{si}_{si_c}.txt".format(args.r_dir, replay=args.replay,
                                                                                           iters=args.iters,
                                                                                           z_dim=args.z_dim,
                                                                                           batch_size=args.batch_size,
                                                                                           replay_batch_size=args.replay_batch_size,
                                                                                           lr=args.lr, aug=args.aug,
                                                                                           main_model=model.name,
                                                                                           train_order=args.dataset_order, seed=args.seed,
                                                                                            val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c),
                "w")
            output_file.write('ADEs:{ades}\nAverage_ADE:{ade}\nFDEs:{fdes}\nAverage_FDE:{fde}'.format(ades=plot_ade_list,ade=metric_dict["average_ade"],fdes=plot_fde_list,fde=metric_dict["average_fde"]))
            output_file.close()

            results_dict = {}
            results_dict["parameters"] = {"iters": args.iters, "z_dim": args.z_dim, "batch_size": args.batch_size, "lr": args.lr}
            results_dict["training order"]= train_order
            results_dict["ade per task"] = plot_ade_list
            results_dict["fde per task"] = plot_fde_list
            results_dict["average ade per task"] = metric_dict["average_ade"]
            results_dict["average fde per task"] = metric_dict["average_fde"]
            utils.save_dict(results_dict, "continual_learning_{replay}_{z_dim}_{batch_size}_{replay_batch_size}_{aug}_{main_model}_{train_order}_{seed}_{val}_{val_class}_{si}_{si_c}".format(replay=args.replay, z_dim=args.z_dim, batch_size=args.batch_size, replay_batch_size=args.replay_batch_size, aug=args.aug, main_model=model.name, train_order=args.dataset_order, seed=args.seed, val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c))


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
