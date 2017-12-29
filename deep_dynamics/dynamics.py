class Dynamics(object):
    def __init__(self):
        pass

    def step(self, dt):
        raise NotImplemented

    def steps_per_epoch(self):
        raise NotImplemented

    @staticmethod
    def setup_parser(self, parser):
        raise NotImplemented
