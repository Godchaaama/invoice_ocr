import cv2
import numpy as np
from deskew import determine_skew
import math
from typing import Union, Tuple

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def auto_morphology(thresh):
    text_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    density = text_pixels / total_pixels
    if density > 0.10:
      ksize = (1, 1)
    elif density > 0.05:
      ksize = (3, 3)
    elif density > 0.01:
      ksize = (5, 5)
    else:
      ksize = (7, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    closed = cv2.erode(dilated, kernel, iterations=2)
    return closed

def preprocess_pipeline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)
    if angle == None:
      angle = 0
    gray = rotate(gray, angle, (0, 0, 0))
    background = cv2.GaussianBlur(gray, (55, 55), 0)
    flattened = cv2.divide(gray, background, scale=255)
    thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    closed = auto_morphology(thresh)
    scaled = cv2.resize(closed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Xóa cv2_imshow vì không khả dụng trong Flask
    return scaled
