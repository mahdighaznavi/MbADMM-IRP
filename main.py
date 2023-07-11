import numpy as np
from numpy import linalg as LA

def smoothed_isotonic_regression(n = 1000, lambda_ = 1, rho = 0.1, num_iters = 100):
    """
    :param n: number of predictors
    :param lambda_: penalty parameter
    :param rho: lagrangian coefficient
    :param num_iters: number of iterations of the algorithm
    """
    # variable initialization
    q = np.zeros((n - 1,))
    p = np.zeros((n - 1,))
    y1 = np.zeros((n - 1,))
    y2 = np.zeros((n - 2,))
    x = np.random.randint(low=0, high=1000, size=n)
    w = np.ones((n,))

    k = 0
    p_next = np.zeros((n - 1,))
    q_next = np.zeros((n - 1,))
    r2_next = np.zeros((n - 2,))

    # dual and primal residuals
    ss, rs = [], []

    while k < num_iters:

        u_next = np.maximum((rho * (q - p) - y1)/(rho + 2 * lambda_), np.zeros_like(p))
        p_next[0] = (2 * w[0] * x[0] + rho * q[0] - rho * u_next[0] - y1[0]) / (2 * w[0] + rho)
        p_next[1:] = (w[1:n-1] * x[1:n-1]+ rho * q[1:] - rho * u_next[1:] - y1[1:] \
                      + rho * q[0:n-2] - y2[0:n-2])/(w[1:n-1] + 2 * rho)
        q_next[:n-2] = (w[1:n-1] * x[1:n-1] + rho * p_next[:n-2] + rho * u_next[:n-2] \
                        + y1[:n-2] + rho * p_next[1:n-1] + y2[:n-2])/(w[1:n-1] + 2 * rho)
        q_next[-1] = (2 * w[-1] * x[-1] + rho * p_next[-1] + u_next[-1])/(2 * w[-1] + rho)
        r1_next = p_next - q_next + u_next
        r2_next = p_next[1:] - q_next[:n-2]
        s1_next = rho * (p_next - q_next - p + q)
        s2_next = rho * (q - q_next)
        r_next = np.sqrt(np.linalg.norm(r1_next)**2 + np.linalg.norm(r2_next)**2)
        s_next = np.sqrt(np.linalg.norm(s1_next)**2 + np.linalg.norm(s2_next)**2)
        y1_next = y1 + rho * r1_next
        y2_next = y2 + rho * r2_next

        p, q, u, y1, y2 = p_next, q_next, u_next, y1_next, y2_next

        k += 1

        ss.append(s_next)
        rs.append(r_next)

    return p, q, u, rs, ss


def multidimensional_ordering(E_1, E_2, Y, W, rho = 1, num_iters = 100):
    """

    :param E_1: representation of edge set E (|E|, n)
    :param E_2: representation of edge set E (|E|, n)
    :param Y: predictors
    :param W: assigned weights
    :return: g, h, v, and list primal and dual residuals
    """
    E_size = E_1.shape[0]
    n = E_1.shape[1]
    y_2 = np.ones(n)
    g = np.ones(n)
    h = np.ones(n)
    y_1 = np.ones(E_size)
    v = np.ones(E_size)
    r = []
    s = []
    for k in range(num_iters):
        v = np.maximum(-np.matmul(E_1, g) + np.matmul(E_2, h) - y_1 / rho, np.zeros_like(y_1))
        new_g = np.matmul(
            LA.inv(
                np.diag(W) +
                rho * np.matmul(E_1.T, E_1) +
                rho * np.identity(n)
            ),
            np.matmul(np.diag(W), Y) +
            rho * np.matmul(np.matmul(E_1.T, E_2), h) -
            rho * np.matmul(E_1.T, v) -
            np.matmul(E_1.T, y_1) +
            rho * h -
            y_2
        )
        new_h = np.matmul(
            LA.inv(
                np.diag(W) +
                rho * np.matmul(E_2.T, E_2) +
                rho * np.identity(n)
            ),
            np.matmul(np.diag(W), Y) +
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
        r.append(np.sqrt(np.power(LA.norm(r_1), 2) + np.power(LA.norm(r_2), 2)))
        s.append(np.sqrt(np.power(LA.norm(s_1), 2) + np.power(LA.norm(s_2), 2) + np.power(LA.norm(s_3), 2)))

        y_1 = y_1 + rho * r_1
        y_2 = y_2 + rho * r_2
        g = np.copy(new_g)
        h = np.copy(new_h)

    return g, h, v, r, s
