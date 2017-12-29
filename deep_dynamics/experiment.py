import numpy as np
import sys

from deep_dynamics.networks.conv2 import Conv2Net
from deep_dynamics.optimizers import PD
from deep_dynamics.utils.running_average import RunningAverage
import torch.optim as optim


class Experiment(object):
    def __init__(self):
        pass

    def update(self, dt):
        raise NotImplemented

    @property
    def steps_per_epoch(self):
        raise NotImplemented

    def run(self, num_epochs, num_average=1, callback=None):
        running_average = RunningAverage(num_average)
        dt = 1.0 / self.steps_per_epoch
        for i in range(num_epochs):
            for j in range(self.steps_per_epoch):
                loss = self.update(dt)
                loss_avg = running_average.add(loss)
                if callback is not None:
                    callback(loss_avg)
                progress = (j + 1) / self.steps_per_epoch
                _log_progress(i + 1, num_epochs, progress, loss_avg)


def _log_progress(epoch, num_epochs, progress, loss):
    progress = min(max(progress, 0.0), 1.0)
    end = '\n' if progress == 1.0 else '\r'
    progress_bar = '=' * int(np.round(60*progress))
    sys.stdout.write('Epoch [{}/{}][{:60}] loss={:<10.2f}{}'.format(epoch, num_epochs, progress_bar, loss, end))


def create_model(model):
    if model == 'conv2':
        return Conv2Net()
    else:
        raise ValueError("Unknown model:", model)


def create_optimizer(optimizer, parameters, steps_per_epoch):
    if 'Adam' in optimizer:
        return optim.Adam(parameters, **optimizer['Adam'])
    elif 'SGD' in optimizer:
        return optim.SGD(parameters, **optimizer['SGD'])
    elif 'PD' in optimizer:
        return PD(parameters, **optimizer['PD'], steps_per_epoch=steps_per_epoch)
    else:
        raise RuntimeError("Unknown optimizer:", optimizer)

