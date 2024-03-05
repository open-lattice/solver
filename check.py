import numpy as np
import pandas as pd
import scipy.interpolate as si
import matplotlib.pyplot as plt


def check():
    T = np.array([[-1, -1, -1, -1, -1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    _T = T.transpose()
    f = np.array([-20, 0, 0, 0, 0, 0])
    _f = _T.dot(f)
    K = np.array(
        [[100, -100, 0, 0, 0, 0], [-100, 200, -100, 0, 0, 0], [0, -100, 200, -100, 0, 0], [0, 0, -100, 200, -100, 0],
         [0, 0, 0, -100, 200, -100], [0, 0, 0, 0, -100, 200]])
    _K = (_T.dot(K)).dot(T)
    print(_K)
    _u = np.linalg.solve(_K, _f)
    print(_u)
    u = T.dot(_u)
    print(u)
    u_sum = u[0] + u[1] + u[2] + u[3] + u[4] + u[5]


if __name__ == "__main__":
    check()
    # df = pd.read_csv('iwp.csv')
    # df = pd.read_csv('frd.csv')
    # df = pd.read_csv('fk.csv')

    # vf = df['VF'].to_numpy()
    # c11 = df['C11'].to_numpy()
    # c12 = df['C12'].to_numpy()
    # c44 = df['C44'].to_numpy()
    # _c11 = si.InterpolatedUnivariateSpline(vf, c11)
    # _c12 = si.InterpolatedUnivariateSpline(vf, c12)
    # _c44 = si.InterpolatedUnivariateSpline(vf, c44)
    # xs = np.arange(0.0, 1.0, 0.01)
    # fig, ax = plt.subplots(figsize=(6.5, 4))
    ##ax.plot(vf, c11, 'o', label='C11')
    ##ax.plot(vf, c12, 'o', label='C12')
    ##ax.plot(vf, c44, 'o', label='C44')
    ##ax.plot(xs, _c11(xs, 1), label="C11'-cubic")
    ##ax.plot(xs, _c12(xs, 1), label="C12'-cubic")
    # ax.plot(xs, _c44(xs, 1), label="C44'-cubic")
    # ax.legend(loc='upper left', ncol=2)
    # plt.title("fk'")
    # plt.grid(True)
    # plt.show()
