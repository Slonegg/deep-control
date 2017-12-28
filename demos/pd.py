import argparse
import numpy as np
import pylab as p
from scipy.integrate import odeint


def dx_dt(x, target, order, kp, kd):
    assert order in (1, 2)
    if order == 1:
        return kp*(target - x)
    else:
        return np.array([kp*(target - x[1]) - kd*x[0], x[0]])


def make_trajectory(order, kp, kd, start, target, step):
    assert order in (1, 2)

    def func(x, t):
        return dx_dt(x, target=target, order=order, kp=kp, kd=kd)
    if order == 1:
        x0 = start
    else:
        x0 = np.array([0.0, start])
    t = np.arange(0.0, step*1000, step)
    x, infodict = odeint(func, x0, t, full_output=True)
    return x.T, t


def plot_trajectory(x, t, order, kp, kd):
    p.figure()
    p.plot(t, x, 'r-', label='trajectory')
    p.grid()
    p.xlabel('t')
    p.ylabel('x')
    p.title('Controller trajectory with order={} kp={:.4f} kd={:.4f}'.format(order, kp, kd))
    p.show()


if __name__ == '__main__':
    # defaults
    order = 1
    start = 1000.0
    target = 0.0
    step = abs(start - target) / 1000
    omega = 0.08

    # simulates system x'' = kp*(x_target - x) - kd*x' or x' = kp*(x_target - x)
    parser = argparse.ArgumentParser(description='PD trajectory generator')
    parser.add_argument('--start', default=start, help='controller starting point')
    parser.add_argument('--target', default=target, help='controller target')
    parser.add_argument('--step', default=step, help='integration step')
    parser.add_argument('--order', default=order, choices=[1, 2], help='order of the differential equation')
    parser.add_argument('--kp', default=omega**2, help='proportional gain')
    parser.add_argument('--kd', default=2*omega, help='derivative gain')
    parser.add_argument('--output', default='pd.txt', help='trajectory output file')
    args = parser.parse_args()

    x, t = make_trajectory(args.order, args.kp, args.kd, args.start, args.target, args.step)
    x = x[order - 1]
    plot_trajectory(x, t, args.order, args.kp, args.kd)
    np.savetxt(args.output, x)
