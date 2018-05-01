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
        image = Pipeline.scale_down(image, 10)
        image = self.binary_filter_model.process_image(image)

        line_finder = LineFinder(image)
        lines = line_finder.find_lines()

        image = self.draw_markers(undistortedImage, lines)

        return image

    def scale_down(image, times):
        width = image.shape[1] // times
        height = image.shape[0] // times
        return cv2.resize(image, (width, height))

    def draw_markers(self, undistortedImage, lines):
        leftPoints = self.project_line(lines[0])
        rightPoints = self.project_line(lines[1])
        image = self.draw_lane(undistortedImage, leftPoints, rightPoints)
        image = self.draw_line(image, leftPoints, (255, 0, 0))
        image = self.draw_line(image, rightPoints, (0, 0, 255))
        return image

    def project_line(self, linePolynomial):
        points = self.interploate_line(linePolynomial)
        points = self.perspective_transform(points)
        return points

    def draw_lane(self, undistortedImage, leftPoints, rightPoints):
        points = np.concatenate((leftPoints, rightPoints[::-1]))
        laneImage = np.zeros_like(undistortedImage)
        cv2.fillPoly(
            img=laneImage,
            pts=[points],
            color=(0, 255, 0))
        undistortedImage = cv2.addWeighted(
            undistortedImage, 1.0, laneImage, 0.3, 0.0)
        return undistortedImage

    def draw_line(self, undistortedImage, points, color):
        lineImage = np.zeros_like(undistortedImage)
        cv2.polylines(
            img=lineImage,
            pts=[points],
            isClosed=False,
            color=color,
            thickness=5)
        undistortedImage = cv2.addWeighted(
            undistortedImage, 1.0, lineImage, 1.0, 0.0)
        return undistortedImage

    def interploate_line(self, linePolynomial):
        polynomial = Pipeline.scaleup_polynomial(linePolynomial, 10)
        points = list()
        for i in range(10):
            y = i * 2220 / 9
            x = np.polyval(polynomial, y) + 5
            points.append((x, y))
        return points

    def scaleup_polynomial(polynomial, times):
        return np.multiply(polynomial, [1.0 / times, 1, times])

    def perspective_transform(self, points):
        return self.bird_view_model.perspective_transform(points)

if __name__ == "__main__":
    pipeline = Pipeline()

    image = cv2.imread("test_images/test1.jpg")
    image = pipeline.process(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
