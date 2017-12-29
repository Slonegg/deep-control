from .experiment import Experiment


class SimpleAttractor(Experiment):
    def __init__(self, start, target, kp, steps_per_epoch):
        self.x = start
        self.target = target
        self.kp = kp
        self._steps_per_epoch = steps_per_epoch

    def dx_dt(self, x):
        return self.kp*(self.target - x)

    def update(self, dt):
        self.x += self.dx_dt(self.x) * dt
        return abs(self.x)

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch
