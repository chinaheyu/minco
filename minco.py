import math
import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded
from collections.abc import Sequence
from typing import Optional


def beta(x: float, N: int, order: int = 0) -> npt.NDArray[float]:
    c = np.zeros(N + 1)
    for i in range(order, N + 1):
        c[i] = np.power(x, i - order) * math.factorial(i) / math.factorial(i - order)
    return c


def create_problem(m: int, s: int, M: int, N: int, D: Sequence[npt.NDArray[float]], T: Sequence[float]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    MM = np.zeros((2 * M * s, 2 * M * s))
    MM[:s, :2 * s] = np.vstack([beta(0.0, N, i) for i in range(s)])
    MM[-s:, -2 * s:] = np.vstack([beta(T[-1], N, i) for i in range(s)])
    for i in range(1, M):
        di = len(D[i])
        MM[s + 2 * s * (i - 1):s + 2 * s * i, 2 * s * (i - 1):2 * s * i] = np.vstack(
            [beta(T[i - 1], N, j) for j in range(di)] +
            [beta(T[i - 1], N, j) for j in range(2 * s - di)]
        )
        MM[s + 2 * s * (i - 1) + di:s + 2 * s * i, 2 * s * i:2 * s * (i + 1)] = -np.vstack(
            [beta(0.0, N, j) for j in range(2 * s - di)]
        )
    b = np.zeros((2 * M * s, m))
    b[:s, :] = D[0]
    b[-s:, :] = D[-1]
    for i in range(1, M):
        di = len(D[i])
        b[s + 2 * s * (i - 1):s + 2 * s * (i - 1) + di, :] = D[i]
    return MM, b


def diagonal_form(a: npt.NDArray[float], lower: int = 1, upper: int = 1) -> npt.NDArray[float]:
    n = a.shape[0]
    ab = np.zeros((upper + lower + 1, n))
    ab[upper, :] = np.diagonal(a)
    for i in range(lower):
        ab[upper + i + 1, :n - i - 1] = np.diagonal(a, -1 - i)
    for i in range(upper):
        ab[upper - i - 1, i + 1:] = np.diagonal(a, 1 + i)
    return ab


def find_piece(t: float, T: Sequence[float]) -> tuple[int, float]:
    i = 0
    sum_ti = 0.0
    while i < len(T) - 1:
        if t - sum_ti < T[i]:
            break
        sum_ti += T[i]
        i += 1
    return i, sum_ti


class MINCO:
    def __init__(self, m: int, s: int, M: int, N: int, D: Sequence[npt.NDArray[float]], T: Sequence[float]) -> None:
        """
        MINCO (Minimum Control) (link: https://doi.org/10.1109/TRO.2022.3160022)
        :param m: Trajectory dimension.
        :param s: Integrators number.
        :param M: Pieces number.
        :param N: Polynomial degree (2 * s - 1).
        :param D: Boundary conditions.
        :param T: Duration of the pieces.
        """

        self.m = m
        self.s = s
        self.M = M
        self.N = N
        self.D = D
        self.T = T

        MM, b = create_problem(m, s, M, N, D, T)
        ab = diagonal_form(MM, 3 * s - 1, 3 * s - 1)
        self.c = solve_banded((3 * s - 1, 3 * s - 1), ab, b)

    def calculate(self, t: float, order: int = 0) -> npt.NDArray[float]:
        i, sum_ti = find_piece(t, self.T)
        return self.c[2 * self.s * i: 2 * self.s * (i + 1), :].T @ beta(t - sum_ti, self.N, order)


class MINCOTrajectory(MINCO):
    def __init__(self, s: int, waypoints: npt.NDArray[float], time_points: npt.NDArray[float], left_boundaries: npt.NDArray, right_boundaries: npt.NDArray) -> None:
        """
        MINCO (Minimum Control) Trajectory (link: https://doi.org/10.1109/TRO.2022.3160022)
        :param s: Integrators number.
        :param waypoints: Waypoints for the trajectory, specified as an n-by-p matrix. n is the dimension of the trajectory, and p is the number of waypoints.
        :param time_points: Time points for the waypoints of the trajectory, specified as a p-element row vector. p is the number of waypoints.
        :param left_boundaries: Left boundaries for the trajectory, specified as an (s-1)-by-n matrix. n is the dimension of the trajectory.
        :param right_boundaries: Right boundaries for the trajectory, specified as an (s-1)-by-n matrix. n is the dimension of the trajectory.
        """
        m = waypoints.shape[0]
        M = waypoints.shape[1] - 1
        N = 2 * s - 1
        D = [
            np.vstack([[waypoints[:, 0]], left_boundaries]),
            *[waypoints[np.newaxis, :, i] for i in range(M - 1)],
            np.vstack([[waypoints[:, M]], right_boundaries]),
        ]
        T = time_points
        super().__init__(m, s, M, N, D, T)


class MinimumAccelerationTrajectory(MINCOTrajectory):
    def __init__(self, waypoints: npt.NDArray[float], time_points: npt.NDArray[float],
                 velocity_boundary: Optional[npt.NDArray] = None) -> None:
        """
        Minimum Acceleration Trajectory
        :param waypoints: Waypoints for the trajectory, specified as an n-by-p matrix. n is the dimension of the trajectory, and p is the number of waypoints.
        :param time_points: Time points for the waypoints of the trajectory, specified as a p-element row vector. p is the number of waypoints.
        :param velocity_boundary: Velocity boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        """
        if velocity_boundary is None:
            velocity_boundary = np.zeros((waypoints.shape[0], 2))
        boundaries = np.stack([velocity_boundary])
        super().__init__(2, waypoints, time_points, boundaries[:, :, 0], boundaries[:, :, 1])


class MinimumJerkTrajectory(MINCOTrajectory):
    def __init__(self, waypoints: npt.NDArray[float], time_points: npt.NDArray[float],
                 velocity_boundary: Optional[npt.NDArray] = None,
                 acceleration_boundary: Optional[npt.NDArray] = None) -> None:
        """
        Minimum Jerk Trajectory
        :param waypoints: Waypoints for the trajectory, specified as an n-by-p matrix. n is the dimension of the trajectory, and p is the number of waypoints.
        :param time_points: Time points for the waypoints of the trajectory, specified as a p-element row vector. p is the number of waypoints.
        :param velocity_boundary: Velocity boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        :param acceleration_boundary: Acceleration boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        """
        if velocity_boundary is None:
            velocity_boundary = np.zeros((waypoints.shape[0], 2))
        if acceleration_boundary is None:
            acceleration_boundary = np.zeros((waypoints.shape[0], 2))
        boundaries = np.stack([velocity_boundary, acceleration_boundary])
        super().__init__(3, waypoints, time_points, boundaries[:, :, 0], boundaries[:, :, 1])


class MinimumSnapTrajectory(MINCOTrajectory):
    def __init__(self, waypoints: npt.NDArray[float], time_points: npt.NDArray[float],
                 velocity_boundary: Optional[npt.NDArray] = None,
                 acceleration_boundary: Optional[npt.NDArray] = None,
                 jerk_boundary_conditions: Optional[npt.NDArray] = None) -> None:
        """
        Minimum Snap Trajectory
        :param waypoints: Waypoints for the trajectory, specified as an n-by-p matrix. n is the dimension of the trajectory, and p is the number of waypoints.
        :param time_points: Time points for the waypoints of the trajectory, specified as a p-element row vector. p is the number of waypoints.
        :param velocity_boundary: Velocity boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        :param acceleration_boundary: Acceleration boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        :param jerk_boundary_conditions: Jerk boundary for the trajectory, specified as an n-by-2 matrix. n is the dimension of the trajectory.
        """
        if velocity_boundary is None:
            velocity_boundary = np.zeros((waypoints.shape[0], 2))
        if acceleration_boundary is None:
            acceleration_boundary = np.zeros((waypoints.shape[0], 2))
        if jerk_boundary_conditions is None:
            jerk_boundary_conditions = np.zeros((waypoints.shape[0], 2))
        boundaries = np.stack([velocity_boundary, acceleration_boundary, jerk_boundary_conditions])
        super().__init__(4, waypoints, time_points, boundaries[:, :, 0], boundaries[:, :, 1])


__all__ = ['MINCO', 'MINCOTrajectory', 'MinimumAccelerationTrajectory', 'MinimumJerkTrajectory', 'MinimumSnapTrajectory']
