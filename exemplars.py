import abc
import torch
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module for a classifier that can store and use exemplars'''

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []   #--> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # @abc.abstractmethod
    # def feature_extractor(self, images):
    #     pass
