import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a predictor'''

    def __init__(self):
        super().__init__()

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return  next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass
