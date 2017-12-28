from .dynamical_system import DynamicalSystem
import numpy as np


class HarmonicOscillator(DynamicalSystem):
    def __init__(self, start, target, kp, kd):
        self.x = np.array([0.0, start])
        self.target = target
        self.kp = kp
        self.kd = kd

    def dx_dt(self, x):
        return np.array([self.kp*(self.target - x[1]) - self.kd*x[0], x[0]])

    def step(self, dt):
        self.x += self.dx_dt(self.x) * dt
        return abs(self.x[1])
