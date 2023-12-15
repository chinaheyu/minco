import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections.abc import Sequence
from scipy.linalg import solve_banded


def beta(x: float, N: int, order: int = 0) -> npt.NDArray[np.floating]:
    c = np.zeros(N + 1)
    for i in range(order, N + 1):
        c[i] = np.power(x, i - order) * math.factorial(i) / math.factorial(i - order)
    return c


def find_interval(t: float, T: npt.NDArray[np.floating]) -> tuple[int, float]:
    i = 0
    sum_ti = 0.0
    while i < len(T) - 1:
        if t - sum_ti < T[i]:
            break
        sum_ti += T[i]
        i += 1
    return i, sum_ti


def diagonal_form(a: npt.NDArray[np.floating], lower: int = 1, upper: int = 1) -> npt.NDArray[np.floating]:
    n = a.shape[0]
    ab = np.zeros((upper + lower + 1, n))

    ab[upper, :] = np.diagonal(a)

    for i in range(lower):
        ab[upper + i + 1, :n - i - 1] = np.diagonal(a, -1 - i)

    for i in range(upper):
        ab[upper - i - 1, i + 1:] = np.diagonal(a, 1 + i)

    return ab


class MINCO:
    def __init__(self, m: int, s: int, M: int, N: int, D: Sequence[npt.NDArray[np.floating]], T: npt.NDArray[np.floating]) -> None:
        """
        MINCO (Minimum Control) Trajectory Class (link: https://doi.org/10.1109/TRO.2022.3160022)
        :param m: trajectory dimension (integrator chains number)
        :param s: integrators per chain
        :param M: pieces number
        :param N: polynomial degree (2 * s - 1)
        :param D: boundary conditions
        :param T: duration of the pieces
        """

        assert m > 0 and s > 0 and M > 1 and N == 2 * s - 1
        assert len(T.shape) == 1 and len(T) == M
        assert len(D) == M + 1
        assert len(D[0].shape) == 2 and D[0].shape[0] == s and D[0].shape[1] == m
        assert len(D[-1].shape) == 2 and D[-1].shape[0] == s and D[-1].shape[1] == m
        assert all([len(D[i].shape) == 2 and D[i].shape[0] <= s and D[i].shape[1] == m for i in range(1, M)])

        self.m = m
        self.s = s
        self.M = M
        self.N = N
        self.D = D
        self.T = T

        MM, b = self.generate_problem()
        ab = diagonal_form(MM, 3 * s - 1, 3 * s - 1)
        self.c = solve_banded((3 * s - 1, 3 * s - 1), ab, b)

    def generate_problem(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        MM = np.zeros((2 * self.M * self.s, 2 * self.M * self.s))
        MM[:self.s, :2 * self.s] = np.row_stack([beta(0.0, self.N, i) for i in range(self.s)])
        MM[-self.s:, -2 * self.s:] = np.row_stack([beta(self.T[-1], self.N, i) for i in range(self.s)])
        for i in range(1, self.M):
            di = len(self.D[i])
            MM[self.s + 2 * self.s * (i - 1):self.s + 2 * self.s * i, 2 * self.s * (i - 1):2 * self.s * i] = np.row_stack([beta(self.T[i - 1], self.N, j) for j in range(di)] + [beta(self.T[i - 1], self.N, j) for j in range(2 * self.s - di)])
            MM[self.s + 2 * self.s * (i - 1) + di:self.s + 2 * self.s * i, 2 * self.s * i:2 * self.s * (i + 1)] = -np.row_stack([beta(0.0, self.N, j) for j in range(2 * self.s - di)])
        b = np.zeros((2 * self.M * self.s, self.m))
        b[:self.s, :] = self.D[0]
        b[-self.s:, :] = self.D[-1]
        for i in range(1, self.M):
            di = len(self.D[i])
            b[self.s + 2 * self.s * (i - 1):self.s + 2 * self.s * (i - 1) + di, :] = self.D[i]
        return MM, b

    def calculate(self, t: float, order: int = 0) -> npt.NDArray[np.floating]:
        i, sum_ti = find_interval(t, self.T)
        return self.c[2 * self.s * i: 2 * self.s * (i + 1), :].T @ beta(t - sum_ti, self.N, order)

    def plot_trajectory(self):
        fig, axes = plt.subplots(2 * self.s, figsize=(7.16, 3.5 * self.s))
        t = np.linspace(0.0, self.T.sum())
        for i in range(2 * self.s):
            p = np.column_stack([self.calculate(tt, i) for tt in t])
            for j in range(p.shape[0]):
                axes[i].plot(t, p[j, :])
            axes[i].set_title(f'{i}-th derivative')
        for i in range(self.s):
            points = []
            for j in range(len(self.D)):
                for k in range(self.D[j].shape[1]):
                    if self.D[j].shape[0] > i:
                        points.append((self.T[:j].sum(), self.D[j][i, k]))
                points = list(zip(*points))
                axes[i].scatter(points[0], points[1], marker='o', facecolors='none', edgecolors='r')
        fig.tight_layout()
        fig.legend([i for i in range(self.m)])
        fig.show()


def demo():
    minco = MINCO(
        2,
        3,
        3,
        5,
        [
            np.array([
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0]]
            ),
            np.array([
                [1.0, -1.0],
                [1.0, -1.0]
            ]),
            np.array([
                [-1.0, 2.0]
            ]),
            np.array([
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ])
        ],
        np.array([1.0, 1.0, 1.0])
    )
    minco.plot_trajectory()


if __name__ == '__main__':
    demo()
