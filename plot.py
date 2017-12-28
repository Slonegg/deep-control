import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from systems import SimpleAttractor, HarmonicOscillator


class Plotter(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.artists = []
        plt.show(False)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

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


def integrate(plotter, system, step=1.0, num_steps=1000, draw_every=10, color='red'):
    trajectory, = plt.plot([], [], 'r-', color=color, label=type(system).__name__)
    plotter.add_artist(trajectory)

    t = []
    x = []
    for i in range(num_steps):
        # integrate
        if len(t) == 0:
            t = [step]
        else:
            t.append(t[-1] + step)
        x.append(system.step(step))

        # draw trajectory
        if i % draw_every == 0:
            trajectory.set_data(t, x)
            plotter.ax.set_ylim(0, np.amax(x))
            plotter.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot various dynamical system behaviours")
    parser.add_argument("config", default="./plots/simple_attractor_vs_harmonic_oscillator.json",
                        help="path to config file with plot definition")
    parser.add_argument("--step", default=1.0,
                        help="integration step")
    parser.add_argument("--num_steps", default=1001,
                        help="number of steps (batches)")
    args = parser.parse_args()

    # create plotter
    step = 1.0
    plotter = Plotter()
    plotter.ax.set_xlim(0, args.step*args.num_steps)

    colors = ['red', 'green', 'blue']
    color_id = 0

    # draw trajectories
    cfg = json.load(open(args.config))
    systems = [SimpleAttractor, HarmonicOscillator]
    for name, args in cfg.items():
        for s in systems:
            if s.__name__ == name:
                system = s(**args)
                integrate(plotter, system, color=colors[color_id])
                color_id += 1
                del system
                break
        else:
            logging.error('Unknown system:', name)

    plt.show()
