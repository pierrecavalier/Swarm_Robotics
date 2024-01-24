from Tracking.PerformTracking import perform_tracking
from Tracking.PerformTracking import TrackingThread
from glob import glob
from datetime import datetime
import os
import time

# ----------------------------------------------------------------------------------------------------------- Parameters

parallel = False     # a parallel run should not display debug information
debug = 2            # 0 = no debug, 1 = text output, 2 = image processing plots, -1 = show pics after 100 but no debug

soften = 2.0                # Geting this value up will ease detection at a cost of less accurate orientation

roi = (0, 4096, 0, 3000)    # Region Of Interest
fisheye_correction_path = "Tracking/Distortion_data/fisheye.pickle"
results_folder = "Results/Pickles"
video_folder = "Results/Videos"
save_data = True
export_vid = False
video_fps = 15

# KB Radius in px, used in Hough circle detection. A rough estimate can be made through debug=-1
min_KB_radius = 80
max_KB_radius = 95

background_path = "DATA_CLEAN/osc6/fond.bmp"
frames_path = "DATA_CLEAN/osc6"  # frames_path is expected to point to folders containing frames

# -------------------------------------------------------------------------------------------------- Preprocessing paths

background_paths = sorted(glob(background_path, recursive=False))
frames_paths = sorted(glob(frames_path))

f_background_paths = list()
f_frames_paths = list()
f_titles = list()
for i in range(len(background_paths)):
    if "_ignore" not in background_paths[i]:
        f_background_paths.append(background_paths[i])

for i in range(len(frames_paths)):
    if "_ignore" not in frames_paths[i]:
        f_frames_paths.append(frames_paths[i])

for i in range(len(f_background_paths)):
    if "_ignore" not in f_background_paths[i]:
        f_titles.append(f_background_paths[i])

for i in range(len(f_titles)):
    f_titles[i] = f_titles[i].split("/")[-1][:-10]
    f_frames_paths[i] += "/*.png"

assert len(f_background_paths) == len(f_frames_paths) == len(f_titles)

# -------------------------------------------------------------------------------------------------------------- Logging

info = (
    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    str(parallel),
    str(debug),
    str(soften),
    str(roi),
    str(fisheye_correction_path),
    str(results_folder),
    str(video_folder),
    str(save_data),
    str(export_vid),
    str(video_fps),
    str(min_KB_radius),
    str(max_KB_radius),
    background_path,
    frames_path
)


# ------------------------------------------------------------------------------------------------------------- Tracking

owd = os.getcwd()

timer = time.time()

threads = list()
for i in range(len(f_background_paths)):
    if parallel:
        t = TrackingThread(
            roi=roi,
            background_path=f_background_paths[i],
            fisheye_correction_path=fisheye_correction_path,
            frames_path=f_frames_paths[i],
            results_folder=results_folder,
            title=f_titles[i],
            video_fps=video_fps,
            min_radius=min_KB_radius,
            max_radius=max_KB_radius,
            soften=soften
        )
        print(f"Running thread {t.getName()} ...")
        t.start()
        threads.append(t)
    else:
        perform_tracking(
            roi=roi,
            save_data=save_data,
            export_vid=export_vid,
            debug=debug,
            video_folder=video_folder,
            background_path=f_background_paths[i],
            fisheye_correction_path=fisheye_correction_path,
            frames_path=f_frames_paths[i],
            results_folder=results_folder,
            title=f_titles[i],
            video_fps=video_fps,
            min_radius=min_KB_radius,
            max_radius=max_KB_radius,
            soften=soften
        )

for t in threads:
    print(f"Waiting for thread {t.getName()} ...")
    t.join()

with open("./Misc/Tracking_log.txt", "r") as f:
    notice = f.read()
title = "Tracking_log_" + \
        datetime.now().strftime("%m%d") + "_" + \
        datetime.now().strftime("%H%M") + ".txt"
with open(os.path.join("./Logs", title), "w") as f:
    f.write(notice.format(*info, "{:.4f}".format(time.time() - timer)))
print("Done.")
