from collections import deque
from functools import reduce
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import LBFGS


class MeanEstimator(object):
    def __init__(self, error_margin=0.01, min_num_samples=3):
        self._error_margin_squared = error_margin*error_margin
        self._sum_x = None
        self._sum_x_squared = None
        self._num_samples = 0
        self._min_num_samples = min_num_samples

    @property
    def sum(self):
        return self._sum_x

    @property
    def mean(self):
        return self._sum_x / self._num_samples

    @property
    def variance(self):
        n = self._num_samples
        return (self._sum_x_squared - self._sum_x * self._sum_x / n) / (n - 1)

    @property
    def average_variance(self):
        return torch.sum(self.variance) / self._sum_x.size()[0]

    def add_sample(self, x):
        if self._sum_x is None:
            self._sum_x = x.clone()
            self._sum_x_squared = x*x
        else:
            self._sum_x += x
            self._sum_x_squared += x*x
        self._num_samples += 1

        if self._num_samples < self._min_num_samples:
            return False

        num_samples_required = 4 * self.average_variance / self._error_margin_squared
        return self._num_samples > num_samples_required

    def reset(self):
        self._sum_x.fill_(0)
        self._sum_x_squared.fill_(0)
        self._num_samples = 0


class PD(Optimizer):
    def __init__(self, params, steps_per_epoch, k_p, k_d=0.0, target=0.0, eps=1e-8):
        assert k_p > 0 and k_d >= 0
        defaults = dict(k_p=k_p, k_d=k_d, steps_per_epoch=steps_per_epoch, eps=eps, target=target)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PD doesn't support per-parameter options (parameter groups)")
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._avg_loss = None
        self._grad_estimator = MeanEstimator()
        self._sum_grad = None
        self._sum_grad_squared = None
        self._num_samples = 0
        self._loss_ref = None
        self._history_size = steps_per_epoch
        self._loss_history = deque()
        self._grad_history = deque()

    def __setstate__(self, state):
        super().__setstate__(state)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        assert len(self.param_groups) == 1

        loss = closure()
        grad = self._gather_flat_grad()
        group = self.param_groups[0]
        k_p = group['k_p']
        k_d = group['k_d']
        steps_per_epoch = group['steps_per_epoch']
        eps = group['eps']
        target = group['target']

        # gradient*update == gain
        loss_np = np.copy(loss.data.numpy())
        grad_np = np.copy(grad.numpy())
        # if len(self._grad_history) > self._history_size:
        #     g = self._grad_history.popleft()
        #     self._avg_grad -= torch.from_numpy(g)
        #     l = self._loss_history.popleft()
        #     self._avg_loss -= torch.from_numpy(l)
        ok = self._grad_estimator.add_sample(grad)
        if ok:
            self._add_grad(0.15, -self._grad_estimator.mean)
            self._grad_estimator.reset()

        # if self._avg_grad is None:
        #     self._sum_grad = grad
        #     self._sum_grad_squared = grad_squared
        #     self._avg_loss = loss.data
        #     self._loss_ref = loss_np
        # else:
        #     self._sum_grad += grad
        #     self._sum_grad_squared += grad_squared
        #     self._avg_loss += loss.data
        # self._num_samples += 1
        # # self._grad_history.append(grad_np)
        # # self._loss_history.append(loss_np)
        #
        # dt = 1.0 / steps_per_epoch
        #
        # var = (self._sum_grad_squared − (self._sum_grad*self._sum_grad) / self._num_samples) / (self._num_samples − 1)
        #
        # gain = (target - loss.data) * k_p
        # len2 = torch.sum(self._avg_grad*self._avg_grad)
        # if len2 > (gain*gain).numpy():
        #     self._add_grad(0.05, -self._avg_grad)
        #     self._avg_grad = None
        #gain1 = (torch.from_numpy(self._loss_ref) - self._avg_loss) * 2
        #update = self._avg_grad * gain0 / max(torch.sum(self._avg_grad*self._avg_grad), eps)

        #self._loss_ref += gain0.numpy() * self._loss_ref * dt

        return loss.data #self._loss_ref #self._avg_loss[0] / len(self._loss_history)


def where(cond, x, y):
    fcond = cond.type(torch.FloatTensor)
    return fcond * x + (1 - fcond) * y
