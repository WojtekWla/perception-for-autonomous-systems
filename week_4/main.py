import numpy as np
import cv2
import glob

nb_vertical, nb_horizontal = 9,6
objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1,2)

objpoints = []
img_points_left, img_points_right = [], []

images_left = sorted(glob.glob('./rs/left*.png'))
images_right = sorted(glob.glob('./rs/right*.png'))

criteria = criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

assert len(images_left) == len(images_right)
processed = 0

for img_left, img_right in zip(images_left, images_right):
    img_left = cv2.imread(img_left)
    img_right = cv2.imread(img_right)

    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    retl, cornersl = cv2.findChessboardCorners(gray_l, (nb_vertical, nb_horizontal))
    retr, cornersr = cv2.findChessboardCorners(gray_r, (nb_vertical, nb_horizontal))

    if retl and retr:
        cornersl = cv2.cornerSubPix(gray_l, cornersl, (10, 10), (-1, -1), criteria)
        cornersr = cv2.cornerSubPix(gray_r, cornersr, (10,10), (-1,-1), criteria)

        objpoints.append(objp)
        img_points_left.append(cornersl)
        img_points_right.append(cornersr)

        img_left = cv2.drawChessboardCorners(img_left, (nb_vertical, nb_horizontal), cornersl, retl)
        img_right = cv2.drawChessboardCorners(img_right, (nb_vertical, nb_horizontal), cornersr, retr)

        cv2.imshow("Left image", img_left)
        cv2.imshow("Right image", img_right)

        cv2.waitKey(0)
        processed += 1

cv2.destroyAllWindows()
print(f"Corners found in {processed} image pairs")

img_size = gray_l.shape[::-1]

retl, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, img_points_left, img_size, None, None)
retr, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, img_points_right, img_size, None, None)

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
stereo_flags = cv2.CALIB_FIX_INTRINSIC

ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, img_points_left, img_points_right,
    mtx_l, dist_l, mtx_r, dist_r, img_size,
    criteria=stereo_criteria, flags=stereo_flags
)

h, w = gray_l.shape[:2]
new_camera_matrix_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (w, h), 1, (w,h))
new_camera_matrix_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w, h), 1, (w,h))