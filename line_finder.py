import cv2
import numpy as np

WINDOW_NUM = 6
WINDOW_WIDTH = 15
DEBUG = False

class LineFinder:

    def __init__(self, image):
        self.image = image
        self.imageWhiteMask = image > 0

    def find_lines(self):
        leftBase, rightBase = self.line_bases()
        leftMask = self.extract_line_mask(leftBase)
        rightMask = self.extract_line_mask(rightBase)
        return leftMask, rightMask

    def line_bases(self):
        height = self.image.shape[0] // 2
        histogram = np.sum(self.image[height:,:], axis=0)
        midpoint = histogram.shape[0] // 2
        left = np.argmax(histogram[:midpoint])
        right = np.argmax(histogram[midpoint:]) + midpoint
        return (left, right)

    def extract_line_mask(self, base):
        lineMask = np.zeros_like(self.imageWhiteMask)
        slidingWindows = SlidingWindows(
            self.image.shape, base, WINDOW_NUM, WINDOW_WIDTH)
        for windowMask in slidingWindows:
            windowWhiteMask = self.imageWhiteMask & windowMask
            lineMask |= windowWhiteMask
            slidingWindows.update_base(windowWhiteMask)
            if DEBUG:
                self.draw_window_mask(windowMask)
        return lineMask

    def draw_window_mask(self, windowMask):
        image = self.image.copy()
        image[windowMask & ~self.imageWhiteMask] = 50
        cv2.imshow("image", image)
        cv2.waitKey(0)

class SlidingWindows:

    def __init__(self, imageShape, base, num, width):
        self.imageShape = imageShape
        self.baseX = int(base)
        self.baseY = int(imageShape[0])
        self.num = int(num)
        self.width = int(width)
        self.height = int(imageShape[0] / num)

    def __iter__(self):
        return self

    def __next__(self):
        halfWidth = self.width // 2
        left = self.baseX - halfWidth
        right = self.baseX + halfWidth
        top = self.baseY - self.height
        bottom = self.baseY
        self.baseY = top

        if top < 0:
            raise StopIteration()

        mask = np.zeros(self.imageShape, np.bool_)
        mask[top:bottom,left:right] = True

        return mask

    def update_base(self, windowWhiteMask):
        windowWhiteIndices = windowWhiteMask.nonzero()
        if not SlidingWindows.is_window_empty(windowWhiteIndices):
            windowWhiteMean = np.mean(windowWhiteIndices, axis=1)
            self.baseX = int(windowWhiteMean[1])

    def is_window_empty(windowWhiteIndices):
        return len(windowWhiteIndices[0]) == 0
