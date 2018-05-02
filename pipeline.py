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
        self.offset = 0

    def process(self, image):
        undistortedImage = self.camera_model.undistort(image)
        image = self.bird_view_model.create_bird_view(undistortedImage)
        image = Pipeline.scale_down(image, 10)
        image = self.binary_filter_model.process_image(image)

        line_finder = LineFinder(image)
        lineMasks = line_finder.find_lines()
        linePolynomials = self.fit_lines(lineMasks)

        image = self.draw_lines(undistortedImage, lineMasks)
        image = self.draw_lane(image, linePolynomials)
        image = self.draw_text(image, linePolynomials)

        return image

    def scale_down(image, times):
        width = image.shape[1] // times
        height = image.shape[0] // times
        return cv2.resize(image, (width, height))

    def draw_lines(self, image, lineMasks):
        image = self.draw_line(image, lineMasks[0], (255, 0, 0))
        image = self.draw_line(image, lineMasks[1], (0, 0, 255))
        return image

    def draw_line(self, undistortedImage, lineMask, color):
        image = np.zeros(lineMask.shape + (3,), dtype=np.uint8)
        image[lineMask] = (1, 1, 1)
        color = np.full_like(image, color)
        image = np.multiply(image, color)
        image = cv2.resize(image, (bird_view.WIDTH, bird_view.HEIGHT))
        image = self.bird_view_model.create_perspective_view(image)
        image = cv2.addWeighted(
            undistortedImage, 1.0, image, 1.0, 0.0)
        return image

    def fit_lines(self, lineMasks):
        return (
            self.fit_line(lineMasks[0]),
            self.fit_line(lineMasks[1])
        )

    def fit_line(self, lineMask):
        pixels = lineMask.nonzero()
        points = np.multiply(pixels, 10) + 5 # convert to cm
        polynomial = np.polyfit(points[0], points[1], 2)
        return polynomial

    def project_line(self, linePolynomial):
        points = self.interploate_line(linePolynomial)
        points = self.perspective_transform(points)
        return points

    def draw_lane(self, undistortedImage, linePolynomials):
        leftPoints = self.project_line(linePolynomials[0])
        rightPoints = self.project_line(linePolynomials[1])
        points = np.concatenate((leftPoints, rightPoints[::-1]))
        laneImage = np.zeros_like(undistortedImage)
        cv2.fillPoly(
            img=laneImage,
            pts=[points],
            color=(0, 255, 0))
        undistortedImage = cv2.addWeighted(
            undistortedImage, 1.0, laneImage, 0.3, 0.0)
        return undistortedImage

    def draw_text(self, image, lines):
        radius = self.estimate_radius(lines)
        radius = str(radius) if radius < 2000 else ">2000"
        Pipeline.put_text(image, (100, 100),
            "Curvature Radius: " + radius + "m")
        offset = self.estimate_offset(lines)
        offset = "{} cm {}".format(abs(offset),
            "to the left" if offset > 0 else
            "to the right" if offset < 0 else "")
        Pipeline.put_text(image, (100, 200), "Offset: " + offset)
        return image

    def interploate_line(self, linePolynomial):
        points = list()
        for i in range(10):
            y = i * bird_view.HEIGHT / 9
            x = np.polyval(linePolynomial, y)
            points.append((x, y))
        return points

    def perspective_transform(self, points):
        return self.bird_view_model.perspective_transform(points)

    def estimate_radius(self, lines):
        leftRadius = Pipeline.curvature_radius(lines[0])
        rightRadius = Pipeline.curvature_radius(lines[1])
        radius = (leftRadius + rightRadius) / 2.0
        self.radius = 0.95 * self.radius + 0.05 * radius
        radius = int(self.radius) // 50 * 50
        return radius

    def estimate_offset(self, lines):
        leftLinePosition = np.polyval(lines[0], bird_view.HEIGHT)
        rightLinePosition = np.polyval(lines[1], bird_view.HEIGHT)
        offset = (rightLinePosition + leftLinePosition) / 2.0 - \
            bird_view.WIDTH / 2.0
        self.offset = 0.9 * self.offset + 0.1 * offset
        offset = int(self.offset) // 5 * 5
        return offset

    def curvature_radius(polynomial):
        a = polynomial[0]
        b = polynomial[1]
        c = polynomial[2]
        y = bird_view.HEIGHT
        r = (1.0 + (2.0 * a * y + b) ** 2.0) ** 1.5 / abs(2.0 * a) / 100
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
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cap = cv2.VideoCapture("project_video.mp4")
    while True:
        ret, image = cap.read()
        image = pipeline.process(image)
        cv2.imshow("image", image)
        cv2.waitKey(1)
    cap.release()
