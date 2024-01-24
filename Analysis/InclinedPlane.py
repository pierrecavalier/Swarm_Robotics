import numpy as np
import cv2
from Tracking.Utils import FisheyeCorrection
from Analysis.Utils import get_chunks
from glob import glob
from skimage.morphology import label
from Tracking.Pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
import os


def get_gravity(img_path, degrees=0):
    img = cv2.imread(img_path)

    fisheye_correction_path = "Tracking\\Distortion_data\\fisheye.pickle"
    correction = FisheyeCorrection(fisheye_correction_path)
    roi = (2700, 2900, 700, 2500)

    pipe = (correction.undistort, Pipeline.roi_(*roi), Pipeline.threshold_(50),
            Pipeline.morphology_(cv2.MORPH_OPEN, filter_size=(5, 5)))

    for f in pipe:
        img = f(img)

    lbl, num = label(img, return_num=True, connectivity=2)

    centers = list()
    for k in range(1, num + 1):
        x, y, z = np.where(lbl == k)
        centers.append((x.mean(), y.mean()))

    gravity = np.array(centers[0]) - np.array(centers[1])
    gravity /= np.linalg.norm(gravity)
    return gravity


def export_trajectories(data_path, export_path="./Results/Figures/"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    # dim 0 = frame #, dim 1 = kb #, dim 2 = [[px, py], array(direction)]

    chunks, _ = get_chunks(data)

    title = data_path.split("/")[-1][:-8]

    chunks_l = list()
    chunks_r = list()
    for k in range(len(chunks)):
        if chunks[k][0, 1] > 1500:
            ch = chunks[k] - chunks[k][0, :]
            ch[:, 1] = -ch[:, 1]
            chunks_l.append(ch)
        else:
            chunks_r.append(chunks[k] - chunks[k][0, :])

    plt.subplot(211)

    plt.title(title)
    plt.annotate("",
                 xy=(0.75, 0.8), xycoords='figure fraction',
                 xytext=(0.85, 0.8), textcoords='figure fraction',
                 arrowprops=dict(arrowstyle="->", facecolor='black', edgecolor='black',
                                 connectionstyle="arc3"),
                 )
    for id in range(len(chunks_l)):
        plt.plot(chunks_l[id][:, 0], chunks_l[id][:, 1])
    plt.subplot(212)
    for id in range(len(chunks_r)):
        plt.plot(chunks_r[id][:, 0], chunks_r[id][:, 1])

    plt.savefig(os.path.join(export_path, title+ "_raw.png"))
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    os.chdir("..")
    background_paths = [
        "DATA_RAW\\Incl.Plane_KB6\\deg00\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg05\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg10\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg15\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg20\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg25\\back.bmp",
        "DATA_RAW\\Incl.Plane_KB6\\deg30\\back.bmp"
    ]

    gravities = dict()
    for back in background_paths:
        gravities[back.split("\\")[2]] = get_gravity(back)

    pickles = glob("../Results/Pickles/Incl.Plane_KB6/**/*.pickle", recursive=True)
    print(gravities)
