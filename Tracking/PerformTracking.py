from glob import glob
import cv2
from Tracking.Pipeline import Pipeline, BackgroundFiltering
from Tracking.ImageProcessing import ImageProcessing, Tracking
from Tracking.Utils import FisheyeCorrection
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import subprocess
from threading import Thread


class TrackingThread(Thread):
    """
    Performs tracking via an individual thread.
    Can't export video, or debug, because writing on disc with threads is too much effort.
    """
    def __init__(self,
                 roi=None,
                 background_path=None,
                 fisheye_correction_path=None,
                 frames_path=None,
                 results_folder=None,
                 title="default",
                 video_fps=15,
                 min_radius=90,
                 max_radius=120,
                 soften=0.0):
        super(TrackingThread, self).__init__()
        self.roi=roi
        self.background_path=background_path
        self.fisheye_correction_path=fisheye_correction_path
        self.frames_path=frames_path
        self.results_folder=results_folder
        self.title=title
        self.video_fps=video_fps
        self.min_radius=min_radius
        self.max_radius=max_radius
        self.soften = soften
        return

    def run(self):
        perform_tracking(
            roi=self.roi,
            background_path=self.background_path,
            fisheye_correction_path=self.fisheye_correction_path,
            frames_path=self.frames_path,
            results_folder=self.results_folder,
            video_folder=None,
            save_data=True,
            export_vid=False,
            title=self.title,
            video_fps=self.video_fps,
            debug=0,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            hide_print=True,
            soften=self.soften
        )


def perform_tracking(
        roi=None,
        background_path=None,
        fisheye_correction_path=None,
        frames_path=None,
        results_folder=None,
        video_folder=None,
        save_data=True,
        export_vid=True,
        title="default",
        video_fps=15,
        debug=0,  # 0 = no debug, max.2,
        min_radius=90,
        max_radius=120,
        hide_print=False,
        soften=0.0
):
    # ############################################################################################ Parameters validation

    assert(
        not (roi is None
             or background_path is None
             or fisheye_correction_path is None
             or frames_path is None
             or (results_folder is None and save_data)
             or (video_folder is None and export_vid)
             )
    )

    # #################################################################################################### Frame Loading

    frames = sorted(glob(frames_path))[80:]

    # frames = sorted(glob("./DATA_CLEAN/Wall/KB6_wall0/*.png"))[33:]

    if len(frames) == 0:
        print(f"No files were found at {frames_path}")
        exit(1)

    correction = FisheyeCorrection(fisheye_correction_path)

    background_filter = BackgroundFiltering(background_path)
    background_filter.back = correction.undistort(background_filter.back)
    background_filter.back = Pipeline.roi_(roi[0], roi[1], roi[2], roi[3])(background_filter.back)

    res_destination = os.path.join(results_folder, "\\".join(frames_path.split("\\")[1:-2]))

    #  Mirror structuring directory
    if not os.path.isdir(res_destination):
        os.makedirs(res_destination, exist_ok=True)

    if video_folder is not None:
        video_destination = os.path.join(video_folder, "\\".join(frames_path.split("\\")[1:-2]))
        if not os.path.isdir(video_destination):
            os.makedirs(video_destination, exist_ok=True)

    raw_destination = os.path.join(res_destination, title + "_raw.pickle")
    res_destination = os.path.join(res_destination, title + ".pickle")

    # ######################################################################################################## Pipelines

    preprocessing_pipeline = (
        correction.undistort,
        Pipeline.roi_(roi[0], roi[1], roi[2], roi[3]),
    )

    detection_pipeline = (
        Pipeline.cvt_color_(cv2.COLOR_BGR2GRAY),
        background_filter.apply_subtraction,
        Pipeline.threshold_(90, t_type=cv2.THRESH_TRUNC),
        Pipeline.rescale,
        Pipeline.threshold_(80),
        Pipeline.morphology_(cv2.MORPH_OPEN, filter_size=(5, 5), filter_type=cv2.MORPH_ELLIPSE),
        Pipeline.median_blur_(5),
    )

    # ############################################################################################################# Main

    imp = ImageProcessing(preprocessing_pipeline, detection_pipeline)
    track = Tracking(min_radius, max_radius, soften)

    results = track.tracking_gen(frames, imp, debug=debug)

    if save_data:
        res = list()
        for i in range(len(frames)):
            if debug > 0:
                print("Frame :", frames[i], "---")
            t = time.time()
            res.append(next(results))
            # track.draw_result(cv2.imread(frames[i]),  res[-1])
            if not hide_print:
                print(i + 1, "/", len(frames), "\t", round(time.time() - t, 2), "s.\t",
                      "px/cm :", round(track.cm_to_px, 2))

        with open(res_destination, "wb") as f:
            pickle.dump(res, f)
        if not hide_print:
            print("Done")

    if export_vid:
        with open(res_destination, "rb") as f:
            res = pickle.load(f)

        seed = np.random.randint(0, 2 ** 31 - 1)
        seed_folder = os.path.join(video_folder, f"{seed}/")
        os.mkdir(seed_folder)

        LOADED_FRAMES = list()
        print("\nExporting video ...")
        plt.figure(figsize=(9, 9))
        for i in range(0, len(frames)):
            plt.clf()
            plt.title(title + " : " + str(i))
            if i % 50 == 0:
                maxi = min(50 * (1 + i // 50), len(frames))
                print(f"\tGenerating frames {50 * i // 50} to {maxi} ...")
                LOADED_FRAMES = [imp.preprocess(frames[j]) for j in range(50 * i // 50, maxi)]
            Tracking.draw_result(LOADED_FRAMES[i % 50], res[i], show=False)
            plt.savefig(os.path.join(seed_folder, f"{i:03d}.png"))

        print("Frames generated, exporting video ...")
        # generate video
        os.chdir(seed_folder)
        subprocess.call([
            'ffmpeg', '-framerate', str(video_fps), '-i', '%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'temp.mp4'
        ])
        os.chdir("..")
        if os.path.exists(f"{title}.mp4"):
            os.remove(f"{title}.mp4")
        os.rename(f"{seed}/temp.mp4", f"{title}.mp4")
        print("Cleaning up ...")
        for file_name in glob(f"{seed}/*.png"):
            os.remove(file_name)
        os.rmdir(f"{seed}/")
        print("Done.")
