from deep_dynamics.networks.conv2 import Conv2Net
from deep_dynamics.dynamics import Dynamics
import logging
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from deep_dynamics.optimizers import PD


class Cifar10Training(Dynamics):
    def __init__(self, model, optimizer):
        if model == 'conv2':
            self.net = Conv2Net()
        else:
            raise ValueError("Unknown model:", model)

        self.criterion = nn.CrossEntropyLoss()

        if 'SGD' in optimizer:
            self.optimizer = optim.SGD(self.net.parameters(), **optimizer['SGD'])
        elif 'PD' in optimizer:
            self.optimizer = PD(self.net.parameters(), **optimizer['PD'])
        else:
            raise RuntimeError("Unknown optimizer:", optimizer)

        self.running_loss = 0.0

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.training_set = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.training_loader = DataLoader(self.training_set, batch_size=32, shuffle=True, num_workers=4)

        self.test_set = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False, num_workers=4)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.training_iterator = None
        self.epoch = 0
        self.next_epoch()

    def next_epoch(self):
        self.training_iterator = iter(self.training_loader)
        self.epoch += 1
        logging.info('Epoch [{}] {} batches'.format(self.epoch, self.steps_per_epoch()))

    def step(self, dt):
        try:
            data = next(self.training_iterator)
        except StopIteration:
            self.next_epoch()
            data = next(self.training_iterator)
        inputs, labels = Variable(data[0]), Variable(data[1])

        def closure():
            self.optimizer.zero_grad()
            output = self.net(inputs)
            loss = self.criterion(output, labels)
            loss.backward()
            return loss

        loss = self.optimizer.step(closure)
        return loss.data[0]

    def steps_per_epoch(self):
        return len(self.test_loader)


if __name__ == '__main__':
    trainer = Cifar10Training(model='conv2', optimizer='SGD')
    for i in range(1000):
        trainer.step(1.0)
