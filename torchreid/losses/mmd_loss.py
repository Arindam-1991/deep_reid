from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import numpy as np  #
from scipy.spatial import distance  #
from scipy.stats import norm  #
import matplotlib.pyplot as plt  #
import seaborn as sns  #
import pickle  #
import torch  #
from sklearn.cluster import KMeans #
import random
from torchreid.metrics import compute_distance_matrix

from functools import partial
from torch.autograd import Variable


class MMDLoss(nn.Module):

    def __init__(self, use_gpu=True, batch_size=32, instances=4, global_only=False, distance_only=True, all=False):
        super(MMDLoss, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.instances = instances
        self.global_only = global_only
        self.distance_only = distance_only
        self.all = all

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)
        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_.cuda())
        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return cost

    def mmd_loss(self, source, target):

        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )
        loss_value = self.maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
        # loss_value = loss_value
        return loss_value

    def forward(self, features, camids):

        # Seperating features based on camids
        ucamids = torch.unique(camids, sorted=True)
        ncamids = len(ucamids)
        pos_list = [torch.nonzero(i == camids).squeeze() if torch.numel(torch.nonzero(i == camids)) > 1 else torch.nonzero(i == camids)[0] for i in ucamids]
        split_feat = list(map(lambda pos : features[pos,:].detach(), pos_list))

        # Compute mmd loss of each camera ids w.r.t to all other samples (samples repeated)
        mmd_loss = 0
        for i, f in enumerate(split_feat):
            # print('feature i {} of shape {}'.format(i, f.size()))
            mmd_loss += self.mmd_loss(f, features)
        mmd_loss /= ncamids

        return mmd_loss 
