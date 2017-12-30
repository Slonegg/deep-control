from deep_dynamics.experiment import Experiment, create_model, create_optimizer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


class MoonsTraining(Experiment):
    def __init__(self, model, optimizer, steps_per_epoch=100):
        # setup model
        self.net = create_model(model)

        # setup dataset
        self.x_train, self.x_test, self.y_train, self.y_test = _load_dataset()
        self.x_train = torch.FloatTensor(self.x_train)
        self.y_train = torch.LongTensor(self.y_train)
        self.x_test = torch.FloatTensor(self.x_test)
        self.y_test = torch.LongTensor(self.y_test)
        self._steps_per_epoch = steps_per_epoch

        # setup optimizer
        self.criterion = CrossEntropyLoss()
        self.optimizer = create_optimizer(optimizer, self.net.parameters(), self.steps_per_epoch)

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    def update(self, dt):
        # train on a whole batch
        inputs, labels = Variable(self.x_train), Variable(self.y_train)

        def closure():
            self.optimizer.zero_grad()
            output = self.net(inputs)
            loss = self.criterion(output, labels)
            loss.backward()
            return loss

        loss = self.optimizer.step(closure)
        return loss.data[0]


def _load_dataset():
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)
    return train_test_split(X, Y, test_size=0.3)


if __name__ == '__main__':
    trainer = MoonsTraining(model='perceptron30', optimizer={'SGD': {'lr': 0.5}})
    trainer.run(5)
