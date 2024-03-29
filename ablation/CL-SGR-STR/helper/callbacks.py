from helper import evaluate
from helper import visual_visdom
import statistics as s


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _sample_cb(log, config, visdom=None, test_datasets=None, sample_size=64, iters_per_task=None):
    '''Initiates function for evaluating samples of generative model.

    [test_datasets]     None or <list> of <Datasets> (if provided, also reconstructions are shown)'''

    def sample_cb(generator, batch, task=1):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        if iteration % log == 0:

            # Evaluate reconstruction-ability of model on [test_dataset]
            if test_datasets is not None:
                # Reconstruct samples from current task
                evaluate.show_reconstruction(generator, test_datasets[task-1], config, size=int(sample_size/2),
                                             visdom=visdom, task=task)

            # Generate samples
            evaluate.show_samples(generator, config, visdom=visdom, size=sample_size,
                                  title="Generated images after {} iters in task {}".format(batch, task))

    # Return the callback-function (except if neither visdom or pdf is selected!)
    return sample_cb if (visdom is not None) else None

def _eval_cb(log, test_datasets, visdom=None, iters_per_task=None, test_size=None, classes_per_task=None,
             scenario="class", summary_graph=True, with_exemplars=False):
    def eval_cb(predictor, batch, task=1):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.precision(predictor, test_datasets, task, iteration, visdom=visdom, summary_graph=summary_graph)

    return eval_cb if (visdom is not None) else None


##------------------------------------------------------------------------------------------------------------------##

################################################
## Callback-functions for calculating metrics ##
################################################


def _metric_cb(log, test_datasets, metrics_dict=None, iters_per_task=None, test_size=None):
    def metric_cb(predictor, batch, task=1):
        iteration = batch if task==1 else (task-1)*iters_per_task + batch
        if iteration % log == 0:
            evaluate.metric_statistics(predictor, test_datasets, task, iteration,
                                       metrics_dict=metrics_dict)

    return metric_cb if (metrics_dict is not None) else None



##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def _solver_val_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''
        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (task is None) else " Task : {}/{} |".format(task, tasks)
            bar.set_description(
                ' <VAL>   | {t_stm} val loss: {loss:.3} |'.format(
                    t_stm=task_stm, loss=loss_dict['loss_val']
                )
            )
            bar.update(1)

        # log the loss of predictor (to visdom)
        if (iteration % log ==0) and (visdom is not None):
            if tasks is None or tasks==1:
                plot_data = [loss_dict['loss_val']]
                names = ['prediction']
            else:
                # weight_new_task = 1. / task if replay else 1.
                weight_new_task = 1
                plot_data = [weight_new_task*loss_dict['loss_val']]
                names = ['pred']
                # if replay:
                #     plot_data += [(1 - weight_new_task) * s.mean(loss_dict['pred_traj_r'])]
                #     names += ['pred - r']
                #     # if model.replay_targets=="hard":
                #     #     plot_data += [(1-weight_new_task)*s.mean(loss_dict['pred_traj_r'])]
                #     #     names += ['pred - r']
                #     # elif model.replay_targets=="soft":
                #     #     pass  # todo add distillation functions
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="Val loss (CL_{})".format(visdom["graph"]), env=visdom["env"], ylable="val loss"
            )

    # Return the callback-function
    return cb

def _solver_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''
        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (task is None) else " Task : {}/{} |".format(task, tasks)
            bar.set_description(
                ' <PREDICTOR>   | {t_stm} training loss: {loss:.3} | training pred_traj: {pred_traj:.3} |'.format(
                    t_stm=task_stm, loss=s.mean(loss_dict['loss_total']), pred_traj=s.mean(loss_dict['pred_traj'])
                )
            )
            bar.update(1)

        # log the loss of predictor (to visdom)
        if (iteration % log ==0) and (visdom is not None):
            if tasks is None or tasks==1:
                plot_data = [s.mean(loss_dict['pred_traj'])]
                names = ['prediction']
            else:
                weight_new_task = 1. / task if replay else 1.
                plot_data = [weight_new_task*s.mean(loss_dict['pred_traj'])]
                names = ['pred']
                if replay:
                    plot_data += [(1 - weight_new_task) * s.mean(loss_dict['pred_traj_r'])]
                    names += ['pred - r']
                    # if model.replay_targets=="hard":
                    #     plot_data += [(1-weight_new_task)*s.mean(loss_dict['pred_traj_r'])]
                    #     names += ['pred - r']
                    # elif model.replay_targets=="soft":
                    #     pass  # todo add distillation functions
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="Train loss (CL_{})".format(visdom["graph"]), env=visdom["env"], ylable="training loss"
            )

    # Return the callback-function
    return cb

def _VAE_loss_cb(log, visdom, model, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else "Task: {}/{} |".format(task, tasks)
            bar.set_description(
                ' <VAE>       | {t_stm} training loss: {loss:.3} | training reconL: {reconL:.3} |'.format(
                    t_stm=task_stm, loss=s.mean(loss_dict['loss_total']), reconL=s.mean(loss_dict['reconL'])
                )
            )
            bar.update(1)

        # log the loss of the solver (to visdom)
        if (iteration % log == 0) and (visdom is not None):
            if tasks is None or tasks==1:
                plot_data = [s.mean(loss_dict['reconL']), s.mean(loss_dict['variatL'])]
                names = ['Recon', 'Variat']
            else:
                weight_new_task = 1. / task if replay else 1.
                plot_data = [weight_new_task*s.mean(loss_dict['reconL']), weight_new_task*s.mean(loss_dict['variatL'])]
                names = ['Recon', 'Variat']
                if replay:
                    plot_data += [(1-weight_new_task)*s.mean(loss_dict['reconL_r']), (1-weight_new_task)*s.mean(loss_dict['variatL_r'])]
                    names += ['Recon - r', 'Variat - r']
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="VAE: loss (CL_{})".format(visdom["graph"]), env=visdom["env"], ylable="training loss"
            )

    # Return the callback-function
    return cb






