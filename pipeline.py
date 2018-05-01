#!/usr/bin/env python3

import bird_view
import binary_filter
import camera_calibration
import cv2
from line_finder import LineFinder
import numpy as np

class Pipeline:

    def __init__(self):
        self.camera_model = camera_calibration.Model()
        self.camera_model.load()
        self.bird_view_model = bird_view.Model()
        self.binary_filter_model = binary_filter.Model()
        self.binary_filter_model.load()

    def process(self, image):
        undistortedImage = self.camera_model.undistort(image)
        image = self.bird_view_model.create_bird_view(undistortedImage)
        image = cv2.resize(image, (97, 222))
        image = self.binary_filter_model.process_image(image)

        line_finder = LineFinder(image)
        lines = line_finder.find_lines()

        image = self.draw_lines(undistortedImage, lines)

        return image

    def draw_lines(self, undistortedImage, lines):
        for linePolynomial in lines:
            self.draw_line(undistortedImage, linePolynomial)
        return undistortedImage

    def draw_line(self, undistortedImage, linePolynomial):
        points = self.interploate_line(linePolynomial)
        points = self.perspective_transform(points)
        cv2.polylines(
            img=undistortedImage,
            pts=[points],
            isClosed=False,
            color=(255, 0, 0),
            thickness=5)
        return undistortedImage

    def interploate_line(self, linePolynomial):
        return np.int32([[500, 1000], [500, 2220]])

    def perspective_transform(self, points):
        return points

if __name__ == "__main__":
    pipeline = Pipeline()

    image = cv2.imread("test_images/test1.jpg")
    image = pipeline.process(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
