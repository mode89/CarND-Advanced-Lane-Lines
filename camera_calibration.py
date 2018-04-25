#!/usr/bin/env python3
import cv2
import glob
import numpy as np

class CameraModel:

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

    def find_perspective_transformation(self):
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

        self.perspectiveMatrix = cv2.getPerspectiveTransform(src, dst)
        self.perspectiveMatrixInv = cv2.getPerspectiveTransform(dst, src)

    def bird_view(self, image):
        image = cv2.undistort(
            image, self.cameraMatrix, self.distortionCoeffs)
        image = cv2.warpPerspective(
            image, self.perspectiveMatrix, (970, 2220))
        return image

    def perspective_view(self, image):
        output = np.zeros((720, 1280), image.dtype)
        output = cv2.warpPerspective(
            src=image,
            M=self.perspectiveMatrix,
            dsize=(1280, 720),
            dst=output,
            flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT)
        return output

    def save(self):
        print("Saving calibration data ...")
        np.savez("camera_model",
            cameraMatrix=self.cameraMatrix,
            distortionCoeffs=self.distortionCoeffs,
            perspectiveMatrix=self.perspectiveMatrix,
            perspectiveMatrixInv=self.perspectiveMatrixInv)

    def load(self):
        print("Loading calibration data ...")
        with np.load("camera_model.npz") as data:
            self.cameraMatrix = data["cameraMatrix"]
            self.distortionCoeffs = data["distortionCoeffs"]
            self.perspectiveMatrix = data["perspectiveMatrix"]
            self.perspectiveMatrixInv = data["perspectiveMatrixInv"]

if __name__ == "__main__":
    camera_model = CameraModel()
    camera_model.calibrate(9, 6)
    camera_model.find_perspective_transformation()
    camera_model.save()
    camera_model.load()

    image = cv2.imread("test_images/test1.jpg")
    image = camera_model.bird_view(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
