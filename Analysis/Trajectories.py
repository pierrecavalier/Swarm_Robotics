import numpy as np
import matplotlib.pyplot as plt
import pickle
from Analysis.Utils import get_chunks
import cv2


def fuse_interp(x, chunks, n, logging=False):
    """Fuse chunks with linear interpolation if frame index between two consecutive chunks is lesseq than n"""
    if len(chunks) <= 1:
        if logging:
            return chunks, f"\t ### There was not enough chunks to interpolate.\n"
        return chunks

    res = [chunks[0]]
    count_interp = 0
    count_total = len(res[-1])
    for cid in range(1, len(chunks)):
        count_total += len(chunks[cid])
        diff = x[cid][0] - x[cid-1][-1] - 1
        if diff <= n:
            points = list()
            for i in range(1, diff+1):
                frac = i * 1.0/(diff+1)
                points.append((1.0 - frac)*res[-1][-1] + frac * chunks[cid][0])
            points = np.array(points)
            res[-1] = np.vstack((res[-1], points, chunks[cid]))
            count_interp += len(points)
        else:
            res.append(chunks[cid])

    ratio = (count_total + count_interp) / count_total
    ratio = np.round(ratio*100, 2)
    if logging:
        return res, f"\t ### {ratio} % dilation due to linear interpolation.\n"
    return res


def remerge_trajectories(chunks_p, chunks_d):
    new_chunks_p = list()
    new_chunks_d = list()
    for cid in range(len(chunks_p)):
        new_chunks_p.append(np.zeros(chunks_p[cid].shape))
        new_chunks_d.append(np.zeros(chunks_d[cid].shape))
        for frame in range(chunks_p[cid].shape[0]):
            for kb in range(chunks_p[cid].shape[1]):
                if frame == 0:
                    new_chunks_p[cid][frame, kb] = chunks_p[cid][frame, kb]
                    new_chunks_d[cid][frame, kb] = chunks_d[cid][frame, kb]
                else:
                    dist = float("inf")
                    arg = 0
                    for kb2 in range(chunks_p[cid].shape[1]):
                        d = np.linalg.norm(chunks_p[cid][frame, kb2] - chunks_p[cid][frame-1, kb])
                        if d < dist:
                            dist = d
                            arg = kb2
                    new_chunks_p[cid][frame, kb] = chunks_p[cid][frame, arg]
                    new_chunks_d[cid][frame, kb] = chunks_d[cid][frame, arg]
    return new_chunks_p, new_chunks_d


def get_vel_and_curvature(chunks, id, radius=160):
    res_vel = list()
    res_curv = list()
    for c in chunks:
        x_t = np.gradient(c[:, id, 0])
        y_t = np.gradient(c[:, id, 1])

        vel = np.array([np.sqrt(x_t[i]**2 + y_t[i]**2) for i in range(x_t.size)])

        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

        res_vel.append(vel.mean())
        res_curv.append(curvature_val.mean())
    return np.array(res_vel)*15 * 2.7 / 160.0, np.array(res_curv)


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


def xor_chunk_pic(pickle_path, pictures_paths, roi):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # print(data)
    # dim 0 = frame #, dim 1 = kb #, dim 2 = [[px, py], array(direction)]

    chunks_p, chunks_d, x = get_chunks(data, get_x=True)

    background = cv2.imread(pictures_paths[0])

    for id in range(len(chunks_p)):
        final_image = cv2.imread(pictures_paths[x[id][0]])
        for i in range(1, len(x[id]), 5):
            im2 = cv2.imread(pictures_paths[x[id][i]])
            mask = cv2.subtract(im2, background)
            _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            im2 = cv2.bitwise_and(im2, im2, mask=mask)
            final_image = cv2.add(final_image, im2)

        for k in range(0, len(chunks_p[id]), 5):
            for kb in range(0, len(chunks_p[id][k])):
                plt.plot(chunks_p[id][k, kb, 1], chunks_p[id][k, kb, 0], "b+")
                plt.arrow(chunks_p[id][k, kb, 1], chunks_p[id][k, kb, 0], 60*np.cos(chunks_d[id][k, kb]),
                          60*np.sin(chunks_d[id][k, kb]), color="b")

        plt.imshow(final_image[roi[2]:roi[3], roi[0]:roi[1]])
        plt.show()


if __name__ == "__main__":
    test = [np.arange(1, 3), np.arange(3, 5), np.arange(5, 7)]
    x = [[1, 2], [5, 6], [12, 13]]
    print(test)
    print(fuse_interp(x, test, 5))
