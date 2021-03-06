from deep_dynamics.experiment import Experiment
import numpy as np


class HarmonicOscillator(Experiment):
    def __init__(self, start, target, kp, kd, steps_per_epoch):
        self.x = np.array([0.0, start])
        self.target = target
        self.kp = kp
        self.kd = kd
        self._steps_per_epoch = steps_per_epoch

    def dx_dt(self, x):
        return np.array([self.kp*(self.target - x[1]) - self.kd*x[0], x[0]])

    def update(self, dt):
        self.x += self.dx_dt(self.x) * dt
        return abs(self.x[1])

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch
