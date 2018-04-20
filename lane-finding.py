import cv2
import glob
import numpy as np

def calibrate_camera(nx, ny):
    images = list()
    image_paths = glob.glob("camera_cal/*.jpg")
    print("Loading calibration images ...")
    for path in image_paths:
        images.append(cv2.imread(path))
    image_size = images[0].shape[1::-1]

    object_points = [(i, j, 0) for i in range(ny) for j in range(nx)]
    all_object_points = list()
    all_image_points = list()

    print("Finding chessboard corners ...")
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image, (nx, ny))
        if ret:
            all_object_points.append(object_points)
            all_image_points.append(corners)

    print("Calculating calibration matrix ...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        np.float32(all_object_points),
        np.float32(all_image_points),
        image_size,
        None,
        None)

    return mtx, dist

def save_calibration_data(cameraMatrix, distortionCoeffs):
    print("Saving calibration data ...")
    np.savez("calibration_data",
        cameraMatrix=cameraMatrix,
        distortionCoeffs=distortionCoeffs)

def load_calibration_data():
    print("Loading calibration data ...")
    with np.load("calibration_data.npz") as data:
        return data["cameraMatrix"], data["distortionCoeffs"]

def undistort_image(image, cameraMatrix, distortionCoeffs):
    print("Undistorting image ...")
    return cv2.undistort(image, cameraMatrix, distortionCoeffs)

def perspective_transformation():
    src = np.float32([
        (560, 475),
        (725, 475),
        (298, 660),
        (1010, 660)
    ])

    dst = np.float32([
        (300, 1000),
        (670, 1000),
        (300, 2220),
        (670, 2220)
    ])

    perspMat = cv2.getPerspectiveTransform(src, dst)
    perspMatInv = cv2.getPerspectiveTransform(dst, src)

    return perspMat, perspMatInv

if __name__ == "__main__":
    cameraMatrix, distortionCoeffs = calibrate_camera(9, 6)

    image = cv2.imread("test_images/test1.jpg")
    image = undistort_image(image, cameraMatrix, distortionCoeffs)
