import abc
from torch import nn

class Replayer(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module for a predictor/generator that can be trained with replay'''

    def __init__(self):
        super().__init__()

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        # --> self.[optim_type]  <str> name of optimizer, relevant if optimizer should be reset for every task
        self.optim_type = "adam"
        # --> self.[optim_list]  <list> if optimizer should be reset after each task, provide list of required <dicts>
        self.optim_list = []
        # Replay: temperature for distillation loss (and whether it should be used)
        self.replay_target = "hard" # hard | soft
        self.KD_temp = 2.

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass
