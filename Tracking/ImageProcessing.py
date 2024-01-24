import numpy as np
import cv2
from skimage.morphology import label
from Tracking.Pipeline import Pipeline
import matplotlib.pyplot as plt


class ImageProcessing:
    """
    Allows the appliance of a pipeline to a list of frames, either at once or via a generator.
    """

    def __init__(self, preprocessing=(Pipeline.identity,), detection=(Pipeline.identity,)):
        self.preprocessing = preprocessing
        self.detection = detection

    def bulk_pipeline(self, frames):
        """ Applies the pipeline to multiple frames at once in a thread pool, returns full list"""
        if type(frames) != list:
            frames = [frames]

        new_frames = list()
        for i in range(len(frames)):
            new_frames.append(self.preprocess(frames[i]))

            for func in self.detection:
                new_frames[i] = Pipeline.apply_step(new_frames[i], func)

        return new_frames

    def preproc_detection(self, frames):
        for i in range(len(frames)):
            preproc = self.preprocess(frames[i])

            img = preproc.copy()
            for func in self.detection:
                img = Pipeline.apply_step(img, func)
            yield preproc, img

    def preproc_gen(self, frames):
        """" No processing generator """
        for i in range(len(frames)):
            img = self.preprocess(frames[i])
            yield img

    def preprocess(self, img):
        # image can either be a path, a ndarray or python list
        if type(img) == str:
            img2 = cv2.imread(img)
            if img is None:
                raise OSError(f"CV2 could not find file {img}")
        elif type(img) == np.ndarray or type(img) == list:
            img2 = img.copy()
        else:
            raise TypeError(f"Frame must be list, ndarray or path to frame. Found {type(img)}")

        for func in self.preprocessing:
            img2 = Pipeline.apply_step(img2, func)
        return img2


class Tracking:
    """
    Perform position, direction and identification tracking of Kilobots inside of exoskeletons.
    """
    def __init__(self, min_radius, max_radius, soften=0.0):
        """
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.cm_to_px = 0
        self.cm_to_px_count = 0

        self.count_frames = 0

        self.old_direction = -1

        self.soften = soften

    def bulk_track(self, frames, original_frames, debug=0, filter_ids=()):
        """ Applies the tracking to multiple frames at once, returns full list """
        if type(frames) != list:
            frames = [frames]

        results = list()
        for i in range(len(frames)):
            results.append(self.track_frame(frames[i], original_frames[i], debug=debug))

        return results

    def tracking_gen(self, frames, img_processing, debug=0):
        """ Tracking performed with a generator """
        if type(frames) != list:
            frames = [frames]

        if debug == -1:
            frames = frames[100:]
            self.count_frames = 100

        image_gen = img_processing.preproc_detection(frames)

        for i in range(len(frames)):
            original, detection = next(image_gen)
            yield self.track_frame(detection, original, debug=debug)

    def track_frame(self, piped_frame, og_frame, debug=0):
        """Track Kilobots and returns the results as a tuple containing three np.ndarray of size
           nb_bots : first one is positions, second is orientations, third is ID."""

        if debug > 2:
            plt.imshow(og_frame)
            plt.show()
        circles = cv2.HoughCircles(piped_frame, cv2.HOUGH_GRADIENT, 1.5, minDist=self.min_radius-5,
                                   param1=255, param2=20,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)

        positions = list()
        directions = list()
        mean_px = piped_frame.mean()

        if debug == -1 and self.count_frames >= 100:
            plt.imshow(piped_frame)
            plt.show()

        self.count_frames += 1
        if circles is not None:

            circles = np.uint16(np.around(circles))[0]

            if len(circles) == 0:
                print("WARNING : no KB found, consider reducing HoughCircles thresholds.")
            if debug > 0:
                print(f"Number of circles found :", len(circles), f"R_0 ={circles[0][2]}")

            for kb in range(len(circles)):
                center_y, center_x, r = circles[kb]

                # types are short by default
                center_x = int(center_x)
                center_y = int(center_y)
                r = int(r)

                self.cm_to_px *= self.cm_to_px_count
                self.cm_to_px += r / 2.75
                self.cm_to_px_count += 1
                self.cm_to_px /= self.cm_to_px_count

                up = np.clip(center_y + r, None, og_frame.shape[1])
                dup = center_y + r - up
                down = np.clip(center_y - r, 0, None)
                ddown = down - center_y + r
                right = np.clip(center_x + r, None, og_frame.shape[0])
                dright = center_x + r - right
                left = np.clip(center_x - r, 0, None)
                dleft = left - center_x + r

                kb_img = og_frame[left:right, down:up]
                if len(kb_img.shape) == 3:
                    kb_img = kb_img[:, :, 0]

                # PADDING
                kb_img = np.pad(kb_img, pad_width=((dleft, dright), (ddown, dup)), mode='edge')

                height, width = kb_img.shape

                if debug > 1:
                    plt.title(f"Kilobot {kb} found at {center_x}, {center_y}")
                    plt.imshow(kb_img)
                    plt.show()

                cX = round(width / 2)
                cY = round(height / 2)
                kb_img = Pipeline.invert(kb_img)

                # keep KB disc
                circle_img = np.zeros((height, width), np.uint8)
                cv2.circle(circle_img, (cX, cY), r-10, 1, thickness=-1)
                kb_img = cv2.bitwise_and(kb_img, kb_img, mask=circle_img)

                center_img = Pipeline.threshold_(140)(kb_img)
                center_img = Pipeline.morphology_(cv2.MORPH_OPEN, filter_size=(10, 10))(center_img)

                # plt.imshow(center_img)
                # plt.show()

                center_circles = cv2.HoughCircles(center_img, cv2.HOUGH_GRADIENT,
                                                  1.5, minDist=self.min_radius - 5,
                                                  param1=255, param2=20,
                                                  minRadius=round(self.min_radius*0.57),
                                                  maxRadius=round(self.max_radius*0.60))

                # xx = round(int(center_circles[0][0][0]))
                # yy = round(int(center_circles[0][0][1]))
                # rr = round(int(center_circles[0][0][2]))
                # cv2.circle(kb_img, (xx, yy), rr, 1, thickness=2)
                # plt.imshow(kb_img)
                # plt.show()

                kb_exo = np.zeros((height, width), np.uint8)

                # Extrude center, either by detection of the inside part or by moments, whichever is best
                if center_circles is not None:
                    cX = int(round(center_circles[0][0][0]))
                    cY = int(round(center_circles[0][0][1]))
                else:
                    moments = cv2.moments(kb_img)
                    cX = int(round(moments["m10"] / moments["m00"]))
                    cY = int(round(moments["m01"] / moments["m00"]))

                cv2.circle(kb_exo, (cX, cY), round(self.max_radius * 0.6), 1, thickness=-1)
                kb_exo = 1 - kb_exo

                kb_img = cv2.bitwise_and(kb_img, kb_img, mask=kb_exo)

                kb_img[np.where(kb_img == 0)] = 40
                kb_img = Pipeline.median_blur_(5)(kb_img)
                kb_img = Pipeline.rescale(kb_img)

                if debug > 1:
                    plt.imshow(kb_img)
                    plt.show()
                kb_img = Pipeline.threshold_(200 - round(20*self.soften))(kb_img)
                filt_size = 5 - round(self.soften)
                filt_size = max(1, filt_size)
                kb_img = Pipeline.morphology_(cv2.MORPH_ERODE, filter_size=(filt_size, filt_size))(kb_img)
                kb_img = Pipeline.morphology_(cv2.MORPH_DILATE, filter_size=(18, 18))(kb_img)
                labels = label(kb_img, connectivity=2)

                angs = list()
                masses = list()
                blob_pos = list()

                if debug > 1:
                    plt.plot(cX, cY, "o")
                for i in range(1, len(labels)):
                    x, y = np.where(labels == i)
                    if len(x) == 0 or len(y) == 0:
                        continue
                    posx = y.mean()
                    posy = x.mean()
                    blob_pos.append(np.array([posx, posy]))

                    an = np.arctan2(posy-cY, posx-cX)
                    if an < 0:
                        an = 2*np.pi + an
                    angs.append(an)
                    if debug > 1:
                        plt.plot(posx, posy, "o", label=str(i-1))
                    masses.append(len(x))
                best_score = 6
                best_dir = 0
                for a in range(len(angs)):
                    for b in range(a, len(angs)):
                        if np.linalg.norm(blob_pos[a] - blob_pos[b]) < self.max_radius:
                            continue
                        score = abs(abs(angs[a] - angs[b]) - np.pi)
                        if self.old_direction != -1:
                            score += 0.5 * min(abs(self.old_direction - angs[b]),
                                               abs(self.old_direction - angs[a]))

                        if score < best_score:
                            best_score = score
                            to_a = masses[a] < masses[b]
                            if self.old_direction != -1:
                                if abs(self.old_direction - angs[b]) < abs(self.old_direction - angs[a]):
                                    to_a = False
                                else:
                                    to_a = True
                            if to_a:
                                best_dir = 0.5*(angs[a] + (angs[b] + np.pi) % (2.0*np.pi))
                            else:
                                best_dir = 0.5*(angs[b] + (angs[a] + np.pi) % (2.0*np.pi))

                            if self.old_direction != -1:
                                alternative_direction = (best_dir + np.pi) % (2.0*np.pi)
                                dist_alt = abs((alternative_direction - self.old_direction + np.pi) % (2*np.pi) - np.pi)
                                dist_og = abs((best_dir - self.old_direction + np.pi) % (2*np.pi) - np.pi)
                                # print("ALT", dist_alt, dist_og)
                                if dist_alt < dist_og:
                                    best_dir = alternative_direction
                        # print(blob_pos[a], blob_pos[b], angs[a], angs[b], ":", score)

                if best_score > 5:
                    if debug > 0:
                        print("\tCould not get orientation for kb", kb)
                        if debug > 1:
                            plt.imshow(kb_img)
                            plt.show()
                    continue

                if debug > 1:
                    plt.imshow(kb_img)
                    plt.legend()
                    plt.arrow(cX, cY, 60*np.cos(best_dir), 60*np.sin(best_dir))
                    plt.show()

                center_x += round(cX - width/2)
                center_y += round(cY - height/2)
                positions.append((center_x, center_y))
                directions.append(best_dir)

                self.old_direction = best_dir

                if debug > 0:
                    print(f"\tKilobot found at {center_x}, {center_y} | {round(best_dir, 2)} rad")

        return positions, directions, mean_px

    @staticmethod
    def draw_result(img, res, show=True):
        positions, directions, mean_px = res
        img2 = img.copy()
        plt.imshow(img2)
        for i in range(len(positions)):
            plt.plot(positions[i][1], positions[i][0], "bo")
            plt.arrow(positions[i][1], positions[i][0], 80 * np.cos(directions[i][0]), 80 * np.sin(directions[i][1]), width=5,
                      fc="cyan", ec="cyan", head_width=25)
        if show:
            plt.show()

