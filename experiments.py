from main import multidimensional_ordering
from utils import generate_random_graph
import numpy as np
import matplotlib.pyplot as plt


def plot_residuals(rs, ss, rho, experiment:str):
    plt.plot(rs, marker='+', label='primal residual')
    plt.plot(ss, marker='o', label='dual residual')
    plt.title(experiment + ', rho = ' + str(rho))
    plt.legend()
    plt.show()


n = 200
E_1, E_2 = generate_random_graph(n)
_, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=0.1,
                                          num_iters=100)
plot_residuals(r, s, rho=0.1, experiment='Multi-dimensional Ordering')
_, _, _, r, s = multidimensional_ordering(E_1=E_1, E_2=E_2, Y=1000 * np.random.rand(n), W=np.ones(n), rho=10,
                                          num_iters=10)
plot_residuals(r, s, rho=10, experiment='Multi-dimensional Ordering')
