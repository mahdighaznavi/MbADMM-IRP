from main import multidimensional_ordering, smoothed_isotonic_regression
from utils import generate_random_graph, plot_residuals
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--num_iters', type=int, default=100)
parser.add_argument('--problem', type=str, default='MO', help='MO: multidimensional ordering, SIR: smoothed isotonic regression')
args = parser.parse_args()

n = args.n
rho = args.rho
num_iters = args.num_iters

if args.problem == 'MO':
    E_1, E_2 = generate_random_graph(n)
    _, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=rho,
                                              num_iters=num_iters)
    plot_residuals(r, s, rho=rho, experiment='Multi-dimensional Ordering')
    
    # _, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=0.1,
    #                                           num_iters=100)
    # plot_residuals(r, s, rho=0.1, experiment='Multi-dimensional Ordering')
    # _, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=10,
    #                                           num_iters=10)
    # plot_residuals(r, s, rho=10, experiment='Multi-dimensional Ordering')

elif args.problem == 'SIR':
    _, _, _, r, s = smoothed_isotonic_regression(n=n, rho=rho, num_iters=num_iters)
    plot_residuals(r, s, rho=rho, experiment='Smoothed Isotonic Regression')
else:
    raise 'The problem should be either "MO" or "SIR".'
