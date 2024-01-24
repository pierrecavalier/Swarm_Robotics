import pickle
import numpy as np
import matplotlib.pyplot as plt

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


def load_smooth_trajectory(pickle_path, show=False, show_chunks=False, linear_interp=3, inf_size_limit=12,
                           logging=False, tol=0.3):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # dim 0 = frame #, dim 1 = kb #, dim 2 = [[px, py], array(direction)]

    log = pickle_path + "\n"

    print(pickle_path)
    chunks_p, chunks_d, x, l = get_chunks(data, get_x=True, show_extraction=show_chunks, logging=True, tolerance=tol)
    chunks_p, chunks_d = remerge_trajectories(chunks_p, chunks_d)
    if logging:
        log += l

    log += "\t ### " + str(len(chunks_p)) + " chunks were originally detected.\n"

    assert len(chunks_p) == len(chunks_d)
    k = 2
    kern = np.ones(2 * k + 1) / (2 * k + 1)

    if linear_interp > 0:
        chunks_p, l2 = fuse_interp(x, chunks_p, linear_interp, logging=True)
        if logging:
            log += l2
        chunks_d = fuse_interp(x, chunks_d, linear_interp)

    removed = list()
    for i in range(len(chunks_p)):
        # chunks_p[i] has dim nb_frames x nb_kb x 2
        # chunks_d[i] has dim nb_frames x nb_kb
        if len(chunks_p[i].shape) < 2:
            log += "ERROR at chunk " + str(i) + " ---------------------------------------------------------\n"
            log += "\t Tracking failed to find any Kilobot.\n"
            return -1, -1
        if chunks_p[i].shape[0] - 2 * k < inf_size_limit:
            log += "\t ### -> chunk " + str(i) + " was removed because of size " + str(chunks_p[i].shape[0]) + "\n"
            removed.append(i)
            continue
        new_chunk_p = np.zeros((chunks_p[i].shape[0] - 2 * k, chunks_p[i].shape[1], 2))
        new_chunk_d = np.zeros((chunks_d[i].shape[0] - 2 * k, chunks_p[i].shape[1]))
        for kb in range(chunks_p[i].shape[1]):
            new_chunk_p[:, kb, 0] = np.convolve(chunks_p[i][:, kb, 0], kern, mode='valid')
            new_chunk_p[:, kb, 1] = np.convolve(chunks_p[i][:, kb, 1], kern, mode='valid')
            chunk_cos = np.convolve(np.cos(chunks_d[i][:, kb]), kern, mode='valid')
            chunk_sin = np.convolve(np.sin(chunks_d[i][:, kb]), kern, mode='valid')
            new_chunk_d[:, kb] = np.arctan2(chunk_sin, chunk_cos)
        chunks_p[i] = new_chunk_p
        chunks_d[i] = new_chunk_d

    temp1 = list()
    temp2 = list()
    for i in range(len(chunks_p)):
        if i not in removed:
            temp1.append(chunks_p[i])
            temp2.append(chunks_d[i])
    chunks_p = temp1
    chunks_d = temp2

    if show:
        for cid in range(len(chunks_p)):
            plt.xlim(0, 4096)
            plt.ylim(0, 3000)

            print("\t", len(chunks_p[cid]))
            for k in range(0, len(chunks_p[cid]), 1):
                for kb in range(0, len(chunks_p[cid][k])):
                    plt.plot(chunks_p[cid][k, kb, 1], chunks_p[cid][k, kb, 0], "b+")
                    plt.arrow(chunks_p[cid][k, kb, 1], chunks_p[cid][k, kb, 0], 30 * np.cos(chunks_d[cid][k, kb]),
                              30 * np.sin(chunks_d[cid][k, kb]), color="b")
            plt.show()
            plt.clf()

    log += "\t ### There are now " + str(len(chunks_p)) + " chunks.\n"
    if logging:
        return log
    return chunks_p, chunks_d



def load_traj(path):
    f = open(path,'rb')
    my_chunks = pickle.load(f)
    res = list()
    for chunk in my_chunks:
        x,y,theta = np.array(chunk[0]),np.array(chunk[1]),np.array(chunk[2])
        res.append(np.array([x,y,theta]))
    return res

data = load_traj("./Wall_data/KB5_wall3_xyTh.pickle")

def traj(path):
    a = np.genfromtxt(path,delimiter=",", skip_header=1,missing_values="nan",filling_values=np.NaN)
    return np.array([a[:,2],a[:,3],a[:,4],a[:,0]]) #Tableau x,y,theta,numéro de frame

