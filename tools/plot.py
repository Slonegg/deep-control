import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from deep_dynamics.experiments import Cifar10Training, SimpleAttractor, HarmonicOscillator


class Plotter(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.artists = []
        plt.show(False)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.xlim = 0
        self.ylim = 0

    def add_artist(self, plot):
        self.artists.append(plot)

    def draw(self):
        # clear background
        self.fig.canvas.restore_region(self.background)

        # draw plot
        for a in self.artists:
            self.ax.draw_artist(a)

        # draw legend
        handles, labels = self.ax.get_legend_handles_labels()
        legend_artist = self.ax.legend(handles, labels)
        self.ax.draw_artist(legend_artist)

        # show on screen
        self.fig.canvas.blit(self.ax.bbox)

    def set_limits(self, xlim, ylim):
        self.xlim = max(self.xlim, xlim)
        self.ylim = max(self.ylim, ylim)
        self.ax.set_xlim(0, self.xlim)
        self.ax.set_ylim(0, self.ylim)


def integrate(plotter, system, label, num_epochs=3, num_average=1, draw_every=10, color='red'):
    trajectory, = plt.plot([], [], 'r-', color=color, label=label)
    plotter.add_artist(trajectory)

    class PlotterCallback(object):
        def __init__(self, plotter):
            self.plotter = plotter
            self.time = []
            self.loss = []
            self.step = 0.0

        def __call__(self, loss):
            if self.step % draw_every == 0:
                self.time.append(self.step / system.steps_per_epoch)
                self.loss.append(loss)
                trajectory.set_data(self.time, self.loss)
                self.plotter.set_limits(self.time[-1], np.amax(self.loss))
                self.plotter.draw()
            self.step += 1

    system.run(num_epochs=num_epochs, num_average=num_average, callback=PlotterCallback(plotter))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser("Plot various dynamical system behaviours")
    parser.add_argument("config", default="./plots/simple_attractor_vs_harmonic_oscillator.json",
                        help="path to config file with plot definition")
    parser.add_argument("--num-epochs", default=5, type=int,
                        help="number of epochs to run experiment")
    args = parser.parse_args()

    # create plotter
    step = 1.0
    plotter = Plotter()

    # draw trajectories
    cfg = json.load(open(args.config))
    systems = [Cifar10Training, SimpleAttractor, HarmonicOscillator]
    for plot_entry in cfg:
        for s in systems:
            if s.__name__ in plot_entry.keys():
                system = s(**plot_entry[s.__name__])
                integrate(plotter, system,
                          label=plot_entry['name'],
                          num_epochs=args.num_epochs,
                          num_average=plot_entry.get('average', 1),
                          color=plot_entry['color'])
                del system
                break
        else:
            logging.error('Plot entry not recognized:', plot_entry)

    plt.show()
