from torch.optim.optimizer import Optimizer


class PD(Optimizer):
    def __init__(self, params, steps_per_epoch, k_p, k_d=0.0):
        assert k_p > 0 and k_d >= 0
        defaults = dict(k_p=k_p, k_d=k_d, steps_per_epoch=steps_per_epoch)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure):
        loss = closure()
        for group in self.param_groups:
            k_p = group['k_p']
            k_d = group['k_d']
            steps_per_epoch = group['steps_per_epoch']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p.mul_(-loss.data * k_p / steps_per_epoch)
                p.data.add_(d_p)

        return loss
