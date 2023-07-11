import numpy as np
from numpy import linalg as LA


def multidimensional_ordering(E_1, E_2, Y, W, num_iters, rho = 1):
    """

    :param E_1: representation of edge set E (|E|, n)
    :param E_2: representation of edge set E (|E|, n)
    :param Y: predictors
    :param W: assigned weights
    :return: g, h, and v
    """
    E_size = E_1.shape[0]
    n = E_1.shape[1]
    y_2 = np.ones(n)
    g = np.ones(n)
    h = np.ones(n)
    y_1 = np.ones(E_size)
    v = np.ones(E_size)
    for k in range(num_iters):
        v = np.maximum(-np.matmul(E_1, g) + np.matmul(E_2, h) - y_1 / rho, np.zeros_like(y_1))
        new_g = np.matmul(
            LA.inv(
                W.diagonal() +
                rho * np.matmul(E_1.T, E_1) +
                rho * np.identity(n)
            ),
            np.matmul(W.diagonal(), Y) +
            rho * np.matmul(np.matmul(E_1.T, E_2), h) -
            rho * np.matmul(E_1.T, v) -
            np.matmul(E_1.T, y_1) +
            rho * h -
            y_2
        )
        new_h = np.matmul(
            LA.inv(
                W.diagonal() +
                rho * np.matmul(E_2.T, E_2) +
                rho * np.identity(n)
            ),
            Y.diagonal() +
            rho * np.matmul(np.matmul(E_2.T, E_1), new_g) +
            rho * np.matmul(E_2.T, v) +
            np.matmul(E_2.T, y_1) +
            rho * new_g +
            y_2
        )

        r_1 = np.matmul(E_1, new_g) - np.matmul(E_2, new_h) + v
        r_2 = new_g - new_h
        s_3 = rho * (h - new_h)
        s_2 = np.matmul(E_2, s_3)
        s_1 = rho * np.matmul(E_1, new_g - g) + s_2

        # calculating primal and dual residual
        r = np.sqrt(np.power(LA.norm(r_1), 2) + np.power(LA.norm(r_2), 2))
        s = np.sqrt(np.power(LA.norm(s_1), 2) + np.power(LA.norm(s_2), 2) + np.power(LA.norm(s_3), 2))

        y_1 = y_1 + rho * r_1
        y_2 = y_2 + rho * r_2
        g = np.copy(new_g)
        h = np.copy(new_h)

