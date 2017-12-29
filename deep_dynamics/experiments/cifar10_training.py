from deep_dynamics.experiment import Experiment, create_model, create_optimizer
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class Cifar10Training(Experiment):
    def __init__(self, model, optimizer, num_workers=4):
        # setup model
        self.net = create_model(model)

        # setup dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.training_set = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.training_loader = DataLoader(self.training_set, batch_size=32, shuffle=True, num_workers=num_workers)
        self.test_set = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False, num_workers=num_workers)
        self.training_iterator = iter(self.training_loader)
        self._steps_per_epoch = len(self.training_iterator)

        # setup optimizer
        self.criterion = CrossEntropyLoss()
        self.optimizer = create_optimizer(optimizer, self.net.parameters(), self.steps_per_epoch)

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    def update(self, dt):
        try:
            data = next(self.training_iterator)
        except StopIteration:
            self.training_iterator = iter(self.training_loader)
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


if __name__ == '__main__':
    trainer = Cifar10Training(model='conv2', optimizer='SGD')
    for i in range(1000):
        trainer.step(1.0)
