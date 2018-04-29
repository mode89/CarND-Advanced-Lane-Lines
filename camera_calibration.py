#!/usr/bin/env python3
import cv2
import glob
import numpy as np

class Model:

    def calibrate(self, nx, ny):
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

        self.cameraMatrix = mtx
        self.distortionCoeffs = dist

    def undistort(self, image):
        return cv2.undistort(
            image, self.cameraMatrix, self.distortionCoeffs)

    def save(self):
        print("Saving calibration data ...")
        np.savez("camera_model",
            cameraMatrix=self.cameraMatrix,
            distortionCoeffs=self.distortionCoeffs)

    def load(self):
        print("Loading calibration data ...")
        with np.load("camera_model.npz") as data:
            self.cameraMatrix = data["cameraMatrix"]
            self.distortionCoeffs = data["distortionCoeffs"]

if __name__ == "__main__":
    camera_model = Model()
    camera_model.calibrate(9, 6)
    camera_model.save()
