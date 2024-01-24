import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

# class FisheyeCorrection
# class DistanceConversion


class FisheyeCorrection:
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            dat = pickle.load(f)

        self.DIM = dat["DIM"]
        self.K = np.array(dat["K"], dtype=np.float32)
        self.D = np.array(dat["D"], dtype=np.float32)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)

    def undistort(self, img):
        undistorted_img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return undistorted_img

    @staticmethod
    def calibrate(imgs, data_path):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # #######################################Blob Detector############################################# #

        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 0
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 4000  # minArea may be adjusted to suit for your experiment
        blobParams.maxArea = 12000  # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.8

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.87

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.01

        # Create a detector with the parameters
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)

        ###############################################################################################

        # Arrays to store object points and image points from all the images.


        objpoints = np.array([[[[i, j, 0] for i in range(0, 60, 6) for j in range(0, 60, 6)]]] * len(imgs),
                             dtype=np.float32)  # 3d points
        imgpoints = []  # 2d points in image plane.

        for img in imgs:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints = blobDetector.detect(gray)  # Detect blobs.

            im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            ret, corners = cv2.findCirclesGrid(im_with_keypoints, (10, 10),
                                               flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
                                               blobDetector=blobDetector, parameters=None)  # Find the circle grid

            if ret:
                imgpoints.append(corners)

                # Draw and display the corners.
                im_with_keypoints = cv2.drawChessboardCorners(img, (10, 10), corners, ret)

            plt.imshow(im_with_keypoints)
            plt.show()

        imgpoints = np.array(imgpoints)

        print(imgpoints.shape, objpoints.shape)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

        DIM = imgs[0].shape[::-1][1:]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            DIM,
            K,
            D,
            None,
            None,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        # It's very important to transform the matrix to list.
        data = {'DIM': DIM, 'K': K, 'D': D}
        with open(data_path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    frames = [cv2.imread("Distortion_data/pattern01.png"),
              cv2.imread("Distortion_data/pattern02.png"),
              cv2.imread("Distortion_data/pattern0003.png")]

    path = "Distortion_data/fisheye2.pickle"

    FisheyeCorrection.calibrate(frames, path)
