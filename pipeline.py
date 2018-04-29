#!/usr/bin/env python3

import bird_view
import binary_filter
import camera_calibration
import cv2

if __name__ == "__main__":
    camera_model = camera_calibration.Model()
    camera_model.load()
    bird_view_model = bird_view.Model()
    binary_filter_model = binary_filter.Model()
    binary_filter_model.load()

    image = cv2.imread("test_images/test1.jpg")
    image = camera_model.undistort(image)
    image = bird_view_model.create_bird_view(image)
    image = cv2.resize(image, (97, 222))
    image = binary_filter_model.process_image(image)
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)
