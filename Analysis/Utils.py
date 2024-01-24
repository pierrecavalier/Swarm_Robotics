import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_results(path_to_res, cm_to_px=31):
    pos = list()
    dir = list()
    ids = list()
    with open(path_to_res, "rb") as f:
        temp = pickle.load(f)

    for i in range(len(temp)):
        positions, directions, identifications = temp[i]
        pos.append(positions / cm_to_px)
        dir.append(directions)
        ids.append(identifications)
    print(pos, "\n\n\n", dir, "\n\n\n", ids)
    return np.array(pos, dtype=float), np.array(dir, dtype=float), np.array(ids)


def get_chunks(data, mass_percentile=50, tolerance=0.3, get_x=False, show_extraction=False, logging=False):
    """
    Returns positions_chunks, direction_chunks where
    * positions_chunks is a list of size nb_chunks containing np.arrays of size nb_frames x nb_KB x 2
    * direction_chunks is a list of size nb_chunks containing np.arrays of size nb_frames x nb_KB

    get_x allows to return a third list of size nb_chunks, mapping chunks entries to frame index
    """
    mean_mass = np.array([data[i][2] for i in range(len(data))])
    threshold = np.percentile(mean_mass, mass_percentile) + tolerance

    x = list()

    megatest = list()

    positions_chunks = list()
    direction_chunks = list()
    cut = True

    count_mass = 0
    for k in range(len(data)):

        megatest.append(0)
        if mean_mass[k] > threshold:
            cut = True
            count_mass += 1
            continue
        if len(data[k][0]) > 0:
            megatest[-1] = 1
            if cut:
                positions_chunks.append(list())
                direction_chunks.append(list())
                x.append(list())
                cut = False
            positions_chunks[-1].append(data[k][0])
            direction_chunks[-1].append(data[k][1])
            x[-1].append(k)
        else:
            if not cut:
                cut = True

    ratio = np.count_nonzero(megatest) / (len(megatest)-count_mass)
    ratio = np.round(ratio*100, 2)
    mass_ratio = 1.0 - count_mass / len(megatest)
    mass_ratio = np.round(mass_ratio*100, 2)

    new_chunks = list()
    new_chunks_d = list()
    new_x = list()

    # filtering out and padding in case of different number of Kilobots found between consecutive frames
    avg_nb_kb = 0
    for cid in range(len(positions_chunks)):
        mini_avg = 0
        for frame in range(len(positions_chunks[cid])):
            mini_avg += len(positions_chunks[cid][frame])
        mini_avg /= (len(positions_chunks[cid]))
        avg_nb_kb += mini_avg
    avg_nb_kb /= len(positions_chunks)
    avg_nb_kb = round(avg_nb_kb)
    for cid in range(len(positions_chunks)):
        for frame in range(len(positions_chunks[cid])):
            while len(positions_chunks[cid][frame]) < avg_nb_kb:
                positions_chunks[cid][frame].append((-1, -1))
            if len(positions_chunks[cid][frame]) > avg_nb_kb:
                positions_chunks[cid][frame] = positions_chunks[cid][frame][:avg_nb_kb]
        for frame in range(len(direction_chunks[cid])):
            while len(direction_chunks[cid][frame]) < avg_nb_kb:
                direction_chunks[cid][frame].append(-1)
            if len(direction_chunks[cid][frame]) > avg_nb_kb:
                direction_chunks[cid][frame] = direction_chunks[cid][frame][:avg_nb_kb]

    # casting chunks to np.array
    for k in range(len(positions_chunks)):
        new_chunks.append(np.array(positions_chunks[k]))
        new_x.append(np.array(x[k]))
        new_chunks_d.append(np.array(direction_chunks[k]))

    if show_extraction:
        plt.plot(mean_mass)
        plt.axhline(threshold)
        plt.plot(megatest, "ro")
        plt.show()

    if get_x:

        if logging:
            l = f"\t ### {mass_ratio} % of the data was below the mass threshold.\n" \
                   f"\t ### {ratio} % of true trajectory was successfully extracted.\n"
            return new_chunks, new_chunks_d, new_x, l
        return new_chunks, new_chunks_d, new_x
    return new_chunks, new_chunks_d
