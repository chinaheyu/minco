import matplotlib.pyplot as plt
from minco import MinimumSnapTrajectory
import numpy as np


def main() -> None:
    mj = MinimumSnapTrajectory(
        np.array([[0.0, 1.0, -1.0, 0.0],
                  [1.0, -1.0, 2.0, 1.0]]),
        np.array([1.0, 1.0, 1.0]),
        np.array([[0.0, 0.0],
                  [-1.0, 1.0]]),
        np.array([[5.0, -5.0],
                  [0.0, 0.0]])
    )

    fig, axes = plt.subplots(2 * mj.s, figsize=(7.16, 3.5 * mj.s))
    t = np.linspace(0.0, np.sum(mj.T))
    for i in range(2 * mj.s):
        p = np.column_stack([mj.calculate(tt, i) for tt in t])
        for j in range(p.shape[0]):
            axes[i].plot(t, p[j, :])
        axes[i].set_title(f'{i}-th derivative')
    for i in range(mj.s):
        boundary_points: list[tuple[float, float]] = []
        for j in range(len(mj.D)):
            for k in range(mj.D[j].shape[1]):
                if mj.D[j].shape[0] > i:
                    boundary_points.append((float(np.sum(mj.T[:j])), float(mj.D[j][i, k])))
        points = list(zip(*boundary_points))
        axes[i].scatter(points[0], points[1], marker='o', facecolors='none', edgecolors='r')
    fig.tight_layout()
    fig.legend([str(i) for i in range(mj.m)])
    fig.show()


if __name__ == '__main__':
    main()
