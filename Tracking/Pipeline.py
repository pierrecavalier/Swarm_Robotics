import os
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation, label
import time


class Pipeline:
    """
    Static methods allowing to build a image processing pipeline.

    foo(img)    is to be used as is in the pipeline.
    foo_(...)   returns a function to use in the pipeline.

    Pipeline example : my_pipeline = ( Pipeline.foo, Pipeline.bar, Pipeline.foo_(parameter) )
    """

    @staticmethod
    def apply_step(img, func):
        """Every function applied goes through here, allowing to build or retrieve information."""
        # t = time.time()
        # a = func(img)
        # print(time.time() - t, "\t", func.__name__)
        return func(img)

    @staticmethod
    def add_(pipe1, pipe2):
        """Branch and add the result of two different pipelines using cv2.add"""
        def add(img):
            img2 = img.copy()
            for f in pipe1:
                img2 = Pipeline.apply_step(img2, f)
            for f in pipe2:
                img = Pipeline.apply_step(img, f)
            return cv2.add(img, img2)

        return add

    @staticmethod
    def subtract_(pipe1, pipe2):
        """Branch and subtract the result of two different pipelines using cv2.subtract"""
        def subtract(img):
            img2 = img.copy()
            for f in pipe1:
                img2 = Pipeline.apply_step(img2, f)
            for f in pipe2:
                img = Pipeline.apply_step(img, f)
            return cv2.subtract(img, img2)

        return subtract

    @staticmethod
    def bitwise_and_(pipe1, pipe2):
        """Branch and add the result of two different pipelines using cv2.add"""

        def bitwise_and(img):
            img2 = img.copy()
            for f in pipe1:
                img2 = Pipeline.apply_step(img2, f)
            for f in pipe2:
                img = Pipeline.apply_step(img, f)
            return cv2.bitwise_and(img, img2)

        return bitwise_and

    @staticmethod
    def morphology_(morph_type, filter_size=(3, 3), filter_type=cv2.MORPH_RECT):
        """cv2.morphology"""
        def morphology(img):
            kernel = cv2.getStructuringElement(filter_type, filter_size)

            return cv2.morphologyEx(img, morph_type, kernel)

        morphology.__name__ = Pipeline.fancy_morphology_name(morph_type, filter_size)

        return morphology

    @staticmethod
    def polar(img):
        value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))

        return cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value,
                                      cv2.WARP_FILL_OUTLIERS)

    @staticmethod
    def cvt_color_(color_type):
        """cv2.cvtColor"""
        def cvt_color(img):
            return cv2.cvtColor(img, color_type)

        return cvt_color

    @staticmethod
    def threshold_(value, maxval=255, t_type=cv2.THRESH_BINARY):
        """cv2.threshold"""
        def threshold(img):
            _, binary = cv2.threshold(img, value, maxval, t_type)
            return binary

        return threshold

    @staticmethod
    def adaptative_threshold_(maxval=255, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, t_type=cv2.THRESH_BINARY,
                             blocksize=11, C=2):
        """ todo """

        def adaptative_threshold(img):
            return cv2.adaptiveThreshold(img, maxval, method, t_type, blocksize, C)

        return adaptative_threshold

    @staticmethod
    def mass_filter_(mini, maxi, value=0):
        """high and low thresholds, keep the in-between part"""

        def mass_filter(img):
            lbl, num = label(img, return_num=True, connectivity=2)
            for k in range(1, num + 1):
                x, y = np.where(lbl == k)
                if mini > len(x) or len(x) > maxi:
                    img[x, y] = value
            return img

        return mass_filter

    @staticmethod
    def save_to_(path):
        """Saves the current img to path, with name corresponding to current time."""
        def save_to(img):
            now = datetime.now()
            cv2.imwrite(os.path.join(path, now.strftime("%m%d%Y_%H%M%S" + ".png")), img)
            return img
        return save_to

    @staticmethod
    def laplace_(kernel_size=3):
        def laplace(img):
            dst = cv2.Laplacian(img, cv2.CV_16S, ksize=kernel_size)
            return cv2.convertScaleAbs(dst)
        return laplace

    @staticmethod
    def roi_(wmin, wmax, hmin, hmax):
        def roi(img):
            return img[hmin:hmax, wmin:wmax]

        return roi

    @staticmethod
    def median_blur_(size):
        def median_blur(img):
            return cv2.medianBlur(img, size)

        return median_blur

    @staticmethod
    def ajust_median_(value):
        def ajust_median(img):
            med = np.median(img)
            sub = np.ones(img.shape) * abs(med-value)
            if med-value >= 0:
                return cv2.subtract(img, sub.astype("uint8"))
            return cv2.add(img, sub.astype("uint8"))

        return ajust_median

    @staticmethod
    def identity(img):
        return img

    @staticmethod
    def skimage_dilation(img):
        """Skimage dilation"""
        return dilation(img)

    @staticmethod
    def invert(img):
        """Invert"""
        return 255 - img

    @staticmethod
    def rescale(img):
        """Rescale intensity"""
        cv2.normalize(img, img, 255, 0, norm_type=cv2.NORM_MINMAX)
        return img
        # maxi = 1.0 * img.max()
        # mini = 1.0 * img.min()
        # return np.array(255*(img-mini)/(maxi - mini), dtype=np.uint8)

    @staticmethod
    def display(img):
        """Display the current img"""
        plt.imshow(img, cmap="gray")
        plt.show()
        return img

    @staticmethod
    def fancy_morphology_name(m_type, size):
        d = {
            6: "BLACKHAT",
            3: "CLOSE",
            1: "DILATE",
            0: "ERODE",
            4: "GRADIENT",
            7: "HITMISS",
            2: "OPEN",
            5: "TOPHAT",
        }
        return d[m_type] + "(" + str(size[0]) + ", " + str(size[1]) + ")"



class BackgroundFiltering:
    """
    Background filtering methods to be used in a pipeline.
    """

    def __init__(self, background, softness=20):

        # image can either be a path, a ndarray or python list
        if type(background) == str:
            back = cv2.imread(background)
            if back is None:
                raise OSError(f"File not found : {background}")
        elif type(background) == np.ndarray or type(background) == list:
            back = background.copy()
        else:
            raise TypeError(f"Background must be list, ndarray or path to frame. Found {type(background)}")

        self.back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        self.softness = softness

    def apply_pointwise(self, img):
        """Pointwise background filtering"""
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if abs(int(img[x, y]) - int(self.back[x, y])) < self.softness:
                    img[x, y] *= 0
        return img

    def apply_soft(self, img):
        """Flexible background filtering"""

        # Create a mask of pixels more distant than softness
        mask = img.copy()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if abs(int(img[x, y]) - int(self.back[x, y])) > self.softness:
                    mask[x, y] = 1
                else:
                    mask[x, y] = 0

        # Smooth out the mask by removing disks <5px in diameter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Apply mask
        img = cv2.bitwise_and(img, img, mask=mask)
        return img

    def apply_absolute(self, img):
        """cv2.absdiff"""
        return cv2.absdiff(img, self.back)

    def apply_subtraction(self, img):
        """cv2.absdiff"""
        return cv2.subtract(img, self.back)

    def apply_absolute_np(self, img):
        """np """
        return np.absolute(img - self.back)

    def apply_subtraction_np(self, img):
        """cv2.absdiff"""
        return np.subtract(img, self.back)

