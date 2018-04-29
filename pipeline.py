#!/usr/bin/env python3

import bird_view
import binary_filter
import camera_calibration
import cv2

class Pipeline:

    def __init__(self):
        self.camera_model = camera_calibration.Model()
        self.camera_model.load()
        self.bird_view_model = bird_view.Model()
        self.binary_filter_model = binary_filter.Model()
        self.binary_filter_model.load()

    def process(self, image):
        image = self.camera_model.undistort(image)
        image = self.bird_view_model.create_bird_view(image)
        image = cv2.resize(image, (97, 222))
        image = self.binary_filter_model.process_image(image)
        return image

if __name__ == "__main__":
    pipeline = Pipeline()

    image = cv2.imread("test_images/test1.jpg")
    image = pipeline.process(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
