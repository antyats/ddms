from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *


def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)


default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))


def FID():
    metric = FID(num_features=1, feature_extractor=default_model)
    metric.attach(default_evaluator, "fid")
    y_true = torch.ones(10, 4)
    y_pred = torch.ones(10, 4)
    state = default_evaluator.run([[y_pred, y_true]])
    return state.metrics["fid"]