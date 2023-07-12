from main import multidimensional_ordering, smoothed_isotonic_regression
from utils import generate_random_graph, plot_residuals, plot_time
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=300)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--num_iters', type=int, default=100)
parser.add_argument('--problem', type=str, default='SIR',
                    help='MO: multidimensional ordering, SIR: smoothed isotonic regression')
parser.add_argument('--exp', type=str, default='residual',
                    help='residual: plot residual in each iteration, time: plot runtime per number of samples')
parser.add_argument('--timestep', type=int, default=10, help='timestep for runtime experiment')
args = parser.parse_args()

n = args.n
rho = args.rho
num_iters = args.num_iters

if args.problem == 'MO':
    if args.exp == 'residual':
        E_1, E_2 = generate_random_graph(n)
        _, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=rho,
                                                  num_iters=num_iters)
        plot_residuals(r, s, rho=rho, experiment='Multi-dimensional Ordering')
    elif args.exp == 'time':
        runtime = []
        for i in range(10, n + 1, args.timestep):
            tmp = 0
            repeat = 5
            for j in range(repeat):
                E_1, E_2 = generate_random_graph(i)
                start_time = time.time()
                multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(i), W=np.ones(i), rho=rho,
                                          num_iters=num_iters)
                tmp += time.time() - start_time
            print(i)
            runtime.append(tmp / repeat)
        plot_time(list(range(1, n + 1, args.timestep)), runtime, experiment='Multi-dimensional Ordering')
    else:
        raise 'The experiment should be residual or runtime'

elif args.problem == 'SIR':
    if args.exp == 'residual':
        _, _, _, r, s = smoothed_isotonic_regression(n=n, rho=rho, num_iters=num_iters)
        plot_residuals(r, s, rho=rho, experiment='Smoothed Isotonic Regression')
    elif args.exp == 'time':
        runtime = []
        for i in range(10, n + 1, args.timestep):
            tmp = 0
            repeat = 5
            for j in range(repeat):
                start_time = time.time()
                smoothed_isotonic_regression(n=i, rho=rho, num_iters=num_iters)
                tmp += time.time() - start_time
            runtime.append(tmp / repeat)
            print(i)
        plot_time(list(range(100, n + 1, args.timestep)), runtime, experiment='Smoothed Isotonic Regression')
    else:
        raise 'The experiment should be residual or runtime'

else:
    raise 'The problem should be either "MO" or "SIR".'
