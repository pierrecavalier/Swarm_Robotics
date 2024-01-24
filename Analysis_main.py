import os.path
from glob import glob
from Analysis.Trajectories import xor_chunk_pic, load_smooth_trajectory, get_vel_and_curvature
from Analysis.InclinedPlane import export_trajectories
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import detrend
from datetime import datetime


def assess_quality(result_path, linear_interp=3):
    files = sorted(glob(os.path.join(result_path, "*.pickle")))
    logs = ""
    for f in files:
        logs += load_smooth_trajectory(f, show=False, show_chunks=False, linear_interp=linear_interp, logging=True)
        logs += "\n"
    with open("./Misc/Analysis_log.txt", "r") as f:
        notice = f.read()
    title = "Analysis_log_" + \
            datetime.now().strftime("%m%d") + "_" + \
            datetime.now().strftime("%H%M") + ".txt"
    with open(os.path.join("./Logs", title), "w") as f:
        f.write(notice.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            logs))
    print("Quality analysis log was saved to", os.path.join("./Logs", title))


if __name__ == "__main__":

    # TODO : clean up
    # See trajectories -------------------------------------------------------------------------------------------------
    assess_quality("Results/Pickles/")

    chunks_p, chunks_d = load_smooth_trajectory("Results/Pickles/test2.pickle", show=True, logging=False)

    plt.plot(chunks_d[0], "+")
    plt.show()

    # Work in progress -------------------------------------------------------------------------------------------------

    # files = sorted(glob("Results/Pickles/InclPlane/0*.pickle"))[::-1]
    # plt.style.use("seaborn")
    # for f in files:
    #     xor_chunk_pic(f, sorted(glob("DATA_CLEAN/Wall/KB5_wall0/*.png")), (0, 4096, 0, 3000))
    #     show_trajectories(f, smooth=True)
    #     chunks_p, chunks_d = load_smooth_trajectory(f, show_chunks=False, linear_interp=3)
    #
    #     if chunks_p == -1:
    #         continue
    #
    #     for i in range(len(chunks_p)):
    #         time = np.linspace(0, len(chunks_p[i])*1.0/15.0, num=len(chunks_p[i]))
    #         title = f.split("/")[-1].split(".")[0] + f" - {i}"
    #         plt.title(title + "\n X position over time (px/s)")
    #         plt.plot(time, chunks_p[i][:, 0, 0], "+")
    #         plt.show()
    #         freq = rfftfreq(len(chunks_p[i]), d=1.0/15.0)
    #
    #         plt.title(title + "\n X position over time (px/s) \n FFT")
    #         plt.plot(freq, np.absolute(rfft(chunks_p[i][:, 0, 0] - chunks_p[i][:, 0, 0].mean())), "+")
    #         plt.show()
    #
    #         signal = detrend(chunks_p[i][:, 0, 1] - chunks_p[i][:, 0, 1].mean(), type="linear")
    #         plt.title(title + "\n Y position over time (px/s)")
    #         plt.plot(time, signal, "+")
    #         plt.show()
    #
    #         plt.title(title + "\n Y position over time (px/s) \n FFT")
    #         plt.plot(freq, np.absolute(rfft(signal)))
    #         plt.show()
    #
    #         plt.title(title + "\n Theta over time (rad/s)")
    #         plt.plot(time, chunks_d[i][:, 0], "+")
    #         plt.show()
    #
    #         plt.title(title + "\n Theta over time (rad/s) \n FFT")
    #         plt.plot(freq, np.absolute(rfft(chunks_d[i][:, 0] - chunks_d[i][:, 0].mean())), "+")
    #         plt.show()
    #
    #     chunksp, _ = load_smooth_trajectory(f)
    #     v, c = get_vel_and_curvature(chunksp, 0)
    #     print(v.mean(), c.mean())
