import os
import numpy as np
from glob import glob
import datetime
import time
import cv2


# -------------------------------------------------------------------------------------------------- Cleaning parameters

raw_folder = "DATA_RAW"
clean_folder = "DATA_CLEAN"

# Number of frames, FPS, X_resolution, Y_resolution, Reduction, Exposure_time (ms), Gain(db)
default_params = (3000, 15, 4096, 3000, 1, 15, 12)

# ---------------------------------------------------------------------------------------------------- Folder management

log = ""

raw_videos = set(glob(os.path.join(raw_folder, "**/*.pds"), recursive=True)) - \
             set(glob(os.path.join(raw_folder, "_ignore/**/*.pds"), recursive=True))

#  Mirror directory structure of RAW into CLEAN
for dirpath, dirnames, filenames in os.walk(raw_folder):
    structure = os.path.join(clean_folder, dirpath[len(raw_folder)+1:])
    if not os.path.isdir(structure):
        os.mkdir(structure)
        log += "Folder created : " + structure + "\n"

log += "\n"

# ----------------------------------------------------------------------------------------------------- Frame extraction

timer = time.time()
for new in raw_videos:

    nout, fps, resolx, resoly, nframe, ex_time, gain = default_params

    N = resolx * resoly

    destination = os.path.join(clean_folder, os.path.join(*(new.split(os.path.sep)[1:])))[:-4]

    if not os.path.isdir(destination):
        os.mkdir(destination)
    else:
        print(f"Destination folder detected for {new}. Extraction canceled.")
        log += f"Destination folder detected for {new}. Extraction canceled.\n"
        continue

    print(f"\nLoading file {new} ... (may take a while) ...")
    A = np.fromfile(new, dtype='uint8', sep="", count=(8 + nout * (624 + N)))

    A = A[8:]

    header = 624
    cnt = header
    digit = 4

    print(f"\nExtracting {nout} frames to {destination} ...")
    log += f"\nExtracting {nout} frames to {destination} ...\n"

    for i in range(nout):
        A_ = A[cnt:cnt + N]

        try:
            A_ = A_.reshape([resoly, resolx])
        except ValueError:
            print(f"\nERROR at frame {i} either the values typed are incorrect or the acquisition failed.\n")
            log += f"\tERROR at frame {i}, either there was less frames than expected or the acquisition failed.\n"
            break
        title = f"{os.path.basename(new)[:-4]}_{i:05d}.png"
        cv2.imwrite(os.path.join(destination, title), A_)

        print("\tExtracted :", os.path.join(destination, title))

        log += "Extracted :" + os.path.join(destination, title) + "\n"

        cnt += N + header

    print("\nDone.\n")

# ------------------------------------------------------------------------------------------------------------- Fill log

with open("./Misc/Extraction_log.txt", "r") as f:
    notice = f.read()

title = "Extraction_log_" + \
        datetime.datetime.now().strftime("%m%d") + "_" + \
        datetime.datetime.now().strftime("%H%M") + ".txt"

with open(os.path.join("./Logs", title), "w") as f:
    f.write(notice.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                          "{:.4f}".format(time.time() - timer),
                          log))

print("\nEverything has been extracted.")
