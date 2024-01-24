import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
from Data_analysis.Utils import get_chunks
from glob import glob


# ----------------------------------------------------------------------------------------------------------------------

def cool_physics_plot(x, y):
    plt.xlabel(r"$\tau (s)$")
    plt.ylabel(r"$<r(t+\tau)-r(t)>^2$ (cm)")
    plt.plot([10**-1, 10**0], [10**0, 10**1], "r--")
    plt.annotate(r"$\propto \tau$", (10**-0.2, 10**0.5))
    plt.plot([10**-1, 10**0], [10**0, 10**2], "g--")
    plt.annotate(r"$\propto \tau^2$", (10**-0.2, 10**1.5))
    plt.loglog(x, y, "o")
    plt.show()


def make_msd_plot(
        path = "../Results/Pickles/Incl.Plane/deg0/KB6/6.pickle",
        fps = 15,
        cm_to_px = 31,
        save_to = "",
        title="",
        show=False,
        split=True
):
    seaborn.set_theme()
    full_tau = list()
    full_msd = list()

    positions, directions, ids = get_chunks(path, cm_to_px)
    for position_chunk in positions:
        for kb in range(position_chunk.shape[1]):
            msd_list = list()

            tau = np.arange(1, 15)
            for t in tau:
                diff = np.array([position_chunk[i+t] - position_chunk[i] for i in range(len(position_chunk)-t)])
                diff_sq = diff ** 2

                msd = np.mean(diff_sq)
                msd_list.append(msd)

            tau = tau.astype(float) / fps

            # cool_physics_plot(tau, msd_list)

            full_tau.append(tau)
            full_msd.append(msd_list)
            # plt.plot(tau,  np.sqrt(np.array(msd_list)) / tau)
            # plt.show()

    fig = plt.figure(figsize=(9, 9))

    if len(title) == 0:
        title = path.split("\\")[-1][:-7]

    plt.title(title)

    if split:
        ax = fig.add_subplot(111, frameon=False)
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


    if split:
        for x, y, i in zip(full_tau, full_msd, range(1, len(full_tau)+1)):
            ax1 = fig.add_subplot(len(full_tau), 1, i)
            ax1.plot([10**-1, 10**0], [10**0, 10**1], "r--")
            ax1.annotate(r"$\propto \tau$", (10**-0.2, 10**0.5))
            ax1.plot([10**-1, 10**0], [10**0, 10**2], "g--")
            ax1.annotate(r"$\propto \tau^2$", (10**-0.2, 10**1.5))
            ax1.set_xlim(x[0] / 1.5, 1.5*x[-1])
            ax1.set_ylim(10**-2, 10**3)
            ax1.loglog(x, y, "o")
            mean_vel = np.polyfit(x[2:], np.sqrt(y[2:]), 1)[0]
            plt.title("Velocity : " + str(round(mean_vel, 2)) + " cm/s")
    else:
        x = np.hstack(full_tau)
        y = np.hstack(full_msd)
        plt.plot([10 ** -1, 10 ** 0], [10 ** 0, 10 ** 1], "r--")
        plt.annotate(r"$\propto \tau$", (10 ** -0.2, 10 ** 0.5))
        plt.plot([10 ** -1, 10 ** 0], [10 ** 0, 10 ** 2], "g--")
        plt.annotate(r"$\propto \tau^2$", (10 ** -0.2, 10 ** 1.5))
        plt.xlim(x[0] / 1.5, 1.5 * x[-1])
        plt.ylim(10 ** -2, 10 ** 3)
        plt.loglog(x, y, "o")
        mean_vel = np.polyfit(x[2:], np.sqrt(y[2:]), 1)[0]
        plt.title(title + "\n\nVelocity : " + str(round(mean_vel, 2)) + " cm/s")
        plt.xlabel(r"$\tau (s)$")
        plt.ylabel(r"$<r(t+\tau)-r(t)>^2$ (cm)")

    if split:
        ax.set_xlabel(r"$\tau (s)$")
        ax.set_ylabel(r"$<r(t+\tau)-r(t)>^2$ (cm)")

    if save_to:
        plt.savefig(os.path.join(save_to, f"{title}.png"))
    if show:
        plt.show()


if __name__ == "__main__":
    path = "../Results/Pickles/Incl.Plane/deg0/KB6/6.pickle"
    fps = 15
    cm_to_px = 31

    pickles = glob("../Results/Pickles/Incl.Plane_KB6/**/*.pickle", recursive=True)
    for p in pickles:
        make_msd_plot(p, fps, cm_to_px, "../Results/Figures/Trajectories/MSD", split=False)
