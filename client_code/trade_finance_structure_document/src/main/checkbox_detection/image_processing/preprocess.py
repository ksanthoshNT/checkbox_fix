import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, erosion_kernel, dilation_kernel, binary_threshold):
        self.erosion_kernel = erosion_kernel
        self.dilation_kernel = dilation_kernel
        self.binary_threshold = binary_threshold

    def preprocess(self, img=None,image_path=None,roi = None):
        # img = Image.open(image_path).convert('L')
        # img_arr = np.array(img)

        if image_path:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if roi:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]


        eroded = self._erode(img)
        dilated = self._dilate(eroded)
        binary_image = self._binarize(dilated)
        return binary_image

    def _erode(self, image):
        return cv2.erode(image.copy(), kernel=np.ones(self.erosion_kernel, np.uint8))

    def _dilate(self, image):
        return cv2.dilate(image.copy(), kernel=np.ones(self.dilation_kernel, np.uint8))

    def _binarize(self, image):
        _, binary = cv2.threshold(image, self.binary_threshold, 255, cv2.THRESH_BINARY)
        return binary