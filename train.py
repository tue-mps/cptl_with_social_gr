import torch
from torch import optim
import numpy as np
import tqdm
import copy
from data import data_loader
from continual_learner import ContinualLearner
import utils

def train_cl(args, model, train_datasets, replay_model="none", scenario="class", class_per_task=None, iters=2, batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs= list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, metric_cbs=list()):
    '''
    Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]           <nn.Module> main model to optimize across all tasks
    [train_datasets]  <list> with for each task the training <DataSet>
    [replay_mode]     <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]        <str>, choice from "task", "domain" and "class"
    [iters]           <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]       None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]           <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        training_dataset = train_dataset

        # Find [active_classes]
        active_classes = None   # -> for Domain-IL scenario, always all classes are active

        # Reset state of optimizer(s) for every task (if requested) todo

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1


        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        for batch_index in range(1, iters_to_use):

            # Update # iters left on current data-loader(s) and, if need, create new one(s)
            iters_left -= 1
            if iters_left==0:
                # data_loader = iter(utils.get_data_loader())
                loader = iter(data_loader(args, training_dataset, args.batch_size))

                #NOTE: [train_dataset] is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(loader)


            #-------------Collect data----------------#

            ##------CURRENT BATCH-------##
            out = next(loader)                                     # --> sample training data of current task
            # y = y - class_per_task*(task-1) if scenario="task" else y    # --> ITL: adjust y-targets
            # print('\nout', len(out))
            x_rel, y_rel = out[2].to(device), out[3].to(device)                            # --> transfer them to correct device
            seq_start_end = out[6].to(device)
            loss_mask = out[5].to(device)


            ##-----REPLAYED BATCH------##
            if not Exact and not Generative and not Current:
                x_rel_ = y_rel_ = seq_start_end_ = None     # -> if no replay

            #----Generative / Current Replay----#
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                # x_ = x if Current else previous_generator.sample(out.to(device))  # 64*500
                replay_data_loader = iter(data_loader(args, training_dataset, args.replay_batch_size))
                replay_out = next(replay_data_loader)
                if Current is None:
                    x_rel_ = replay_out[2].to(device)
                else:
                    replay_traj = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device), replay_out[6].to(device))
                    x_rel_ = replay_traj[1]
                    x_ = replay_traj[0]
                    seq_start_end_ = replay_traj[2]
                y_rel_ = previous_model(x_rel_)

                # todo ----> How to generate y_ with encoder.py

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once

            # ----> Train Main model
            if batch_index <= iters:

                # Train the main model with this batch
                loss_dict = model.train_a_batch(x_rel, y_rel, x_rel_=x_rel_, y_rel_=y_rel_, loss_mask=loss_mask, rnt=1./task)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)

            # -----> Train Generator
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, rnt=1./task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)

        # ----> UPON FINISHING EACH TASK...

        # Close progress-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_model == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model







