import numpy as np

def generate_random_graph(n):
    Z = [np.random.rand(2) for i in range(n)]
    E_1 = []
    E_2 = []
    for i in range(n):
        for j in range(n):
            if Z[i][0] <= Z[j][0] and Z[i][1] <= Z[j][1]:
                out_tmp = np.zeros(n)
                out_tmp[i] = 1
                in_tmp = np.zeros(n)
                in_tmp[j] = 1
                E_1.append(out_tmp)
                E_2.append(in_tmp)
    return np.array(E_1), np.array(E_2)