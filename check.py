import numpy as np
import pandas as pd
import scipy.interpolate as si
import matplotlib.pyplot as plt


def check():
    K = np.array(
        [[100, -100, 0, 0, 0, 0], [-100, 200, -100, 0, 0, 0], [0, -100, 200, -100, 0, 0], [0, 0, -100, 200, -100, 0],
         [0, 0, 0, -100, 200, -100], [0, 0, 0, 0, -100, 200]])
    f = np.array([-20, 0, 0, 0, 0, 0])
    #f = np.array([-90, 0, 80, 0, 0, 0])
    T = np.array([[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    Tt = T.transpose()

    _f = Tt.dot(f)
    print("_f:", _f)

    _K = (Tt.dot(K)).dot(T)
    print("_K:", _K)

    _u = np.linalg.solve(_K, _f)
    print("_u:", _u)

    u = T.dot(_u)
    print("u: ", u)
    print("constraint check", u[0] - u[5])


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
