import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a predictor'''

    def __init__(self):
        super().__init__()

        # XdG:
        self.mask_dict = None  # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # -SI:
        self.si_c = 0  # -> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1  # -> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # -EWC:
        self.ewc_lambda = 0  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0  # -> keeps track of number of quadratic loss terms (for "offline EWC")

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda



    @abc.abstractmethod
    def forward(self, x):
        pass
