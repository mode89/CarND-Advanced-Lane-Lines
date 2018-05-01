#!/usr/bin/env python3

import cv2
import numpy as np

class Model:

    def __init__(self):
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

        self.perspective_matrix = cv2.getPerspectiveTransform(src, dst)
        self.perspective_matrix_inv = cv2.getPerspectiveTransform(dst, src)

    def create_bird_view(self, image):
        image = cv2.warpPerspective(
            image, self.perspective_matrix, (970, 2220))
        return image

    def create_perspective_view(self, image):
        output = np.zeros((720, 1280), image.dtype)
        output = cv2.warpPerspective(
            src=image,
            M=self.perspective_matrix,
            dsize=(1280, 720),
            dst=output,
            flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT)
        return output

    def perspective_transform(self, points):
        points = np.float32(points)
        pointNum = points.shape[0]
        points = points.reshape((pointNum, 1, 2))
        points = cv2.perspectiveTransform(
            src=points,
            m=self.perspective_matrix_inv)
        points = points[:,:,:2]
        points = points.reshape((pointNum, 2))
        return np.int64(points)
