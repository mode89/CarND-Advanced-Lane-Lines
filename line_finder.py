import numpy as np

class LineFinder:

    def find_lines(self, image):
        print(self.line_bases(image))

    def line_bases(self, image):
        height = image.shape[0] // 2
        histogram = np.sum(image[height:,:], axis=0)
        midpoint = histogram.shape[0] // 2
        left = np.argmax(histogram[:midpoint])
        right = np.argmax(histogram[midpoint:]) + midpoint
        return (left, right)
