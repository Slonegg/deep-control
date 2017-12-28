from .dynamical_system import DynamicalSystem


class SimpleAttractor(DynamicalSystem):
    def __init__(self, start, target, kp):
        self.x = start
        self.target = target
        self.kp = kp

    def dx_dt(self, x):
        return self.kp*(self.target - x)

    def step(self, dt):
        self.x += self.dx_dt(self.x) * dt
        return abs(self.x)
