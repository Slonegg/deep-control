from functools import reduce
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import LBFGS


class PD(Optimizer):
    def __init__(self, params, steps_per_epoch, k_p, k_d=0.0, target=0.0, eps=1e-8):
        assert k_p > 0 and k_d >= 0
        defaults = dict(k_p=k_p, k_d=k_d, steps_per_epoch=steps_per_epoch, eps=eps, target=target)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PD doesn't support per-parameter options (parameter groups)")
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

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
        group = self.param_groups[0]
        k_p = group['k_p']
        k_d = group['k_d']
        steps_per_epoch = group['steps_per_epoch']
        eps = group['eps']
        target = group['target']

        # gradient*update == gain
        gain = (target - loss.data) * k_p
        grad = self._gather_flat_grad()
        update = grad * gain / max(torch.sum(grad*grad), eps)
        self._add_grad(1.0 / steps_per_epoch, update)

        return loss


def where(cond, x, y):
    fcond = cond.type(torch.FloatTensor)
    return fcond * x + (1 - fcond) * y
