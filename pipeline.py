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
        self.radius = 300

    def process(self, image):
        undistortedImage = self.camera_model.undistort(image)
        image = self.bird_view_model.create_bird_view(undistortedImage)
        image = Pipeline.scale_down(image, 10)
        image = self.binary_filter_model.process_image(image)

        line_finder = LineFinder(image)
        lines = line_finder.find_lines()
        lines = (
            Pipeline.transform_polynomial(lines[0]),
            Pipeline.transform_polynomial(lines[1])
        )

        image = self.draw_markers(undistortedImage, lines)
        image = self.draw_text(image, lines)

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

    def draw_text(self, image, lines):
        radius = self.estimate_radius(lines)
        radius = str(radius) if radius < 2000 else ">2000"
        Pipeline.put_text(image, (100, 100),
            "Curvature Radius: " + radius + "m")
        return image

    def interploate_line(self, linePolynomial):
        points = list()
        for i in range(10):
            y = i * 2220 / 9
            x = np.polyval(linePolynomial, y)
            points.append((x, y))
        return points

    def transform_polynomial(polynomial):
        polynomial = Pipeline.scaleup_polynomial(polynomial, 10)
        polynomial[2] += 5
        return polynomial

    def scaleup_polynomial(polynomial, times):
        return np.multiply(polynomial, [1.0 / times, 1, times])

    def perspective_transform(self, points):
        return self.bird_view_model.perspective_transform(points)

    def estimate_radius(self, lines):
        leftRadius = Pipeline.curvature_radius(lines[0])
        rightRadius = Pipeline.curvature_radius(lines[1])
        radius = (leftRadius + rightRadius) / 2.0
        self.radius = 0.95 * self.radius + 0.05 * radius
        radius = int(self.radius) // 50 * 50
        return radius

    def curvature_radius(polynomial):
        a = polynomial[0]
        b = polynomial[1]
        c = polynomial[2]
        r = (1.0 + (2.0 * a * 2220 + b) ** 2.0) ** 1.5 / abs(2.0 * a) / 100
        return r

    def put_text(image, org, text):
        cv2.putText(
            img=image,
            text=text,
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 255),
            thickness=3)

if __name__ == "__main__":
    pipeline = Pipeline()

    image = cv2.imread("test_images/test1.jpg")
    image = pipeline.process(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
