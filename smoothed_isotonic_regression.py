import numpy as np
import matplotlib.pyplot as plt

def smoothed_isotonic_regression(n=1000, lambda_=1, rho=0.1):
    q = np.zeros((n - 1,))
    p = np.zeros((n - 1,))
    y1 = np.zeros((n - 1,))
    y2 = np.zeros((n - 2,))
    x = np.random.randint(low=0, high=1000, size=n)
    w = np.ones((n,))

    k = 0
    p_next = np.zeros((n-1,))
    q_next = np.zeros((n-1,))
    r2_next = np.zeros((n-2,))

    ss, rs = [], []

    while k < 100:

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

    plt.plot(rs, marker='+', label='primal residual r')
    plt.plot(ss, marker='o', label='dual residual s')
    plt.legend()
    plt.savefig(f'SIR_rho_{rho}.png')
