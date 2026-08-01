import cv2
import numpy as np


def preprocess_image(img):
    """Match simulator preprocessing(TestSimulation.preProcessing ): crop, convert to YUV, blur, resize to 200x66, and normalize."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img.astype(np.float32)


def load_rgb_image(path):
    """Load an image from disk as RGB"""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    # Convert BGR to RGB for simulator compatibility.
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
