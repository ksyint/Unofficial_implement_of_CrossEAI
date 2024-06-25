import numpy as np
import cv2

def generate_bounding_box(grad_cam, guided_backprop, threshold=0.5):
    combined_map = (grad_cam + guided_backprop) / 2
    combined_map = (combined_map - np.min(combined_map)) / (np.max(combined_map) - np.min(combined_map))

    binary_map = combined_map > threshold

    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, x + w, y + h
