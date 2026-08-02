import cv2
import numpy as np


def random_flip(img, steering):
    """Horizontal flip; reverse steering direction."""
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def random_brightness(img):
    """Random brightness via HSV Value channel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.4, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_zoom(img):
    """Random zoom (scale crop then resize back)."""
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.3)
    nh, nw = int(h / scale), int(w / scale)
    y1 = (h - nh) // 2
    x1 = (w - nw) // 2
    cropped = img[y1:y1 + nh, x1:x1 + nw]
    return cv2.resize(cropped, (w, h))


def random_pan(img):
    """Random translation"""
    h, w = img.shape[:2]
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_rotate(img, steering, max_angle=5.0):
    """ Random rotation."""
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Adjust the steering label to match the rotated image.
    steering = steering + (angle * 0.002)
    return img, steering


def augment_image(img, steering, aug_prob=0.5):
    """ Random augmentations with probability aug_prob."""
    if np.random.rand() > aug_prob:
        return img, float(steering)

    img, steering = random_flip(img, steering)

    if np.random.rand() < 0.5:
        img = random_brightness(img)
    if np.random.rand() < 0.5:
        img = random_zoom(img)
    if np.random.rand() < 0.5:
        img = random_pan(img)
    if np.random.rand() < 0.5:
        img, steering = random_rotate(img, steering)

    return img, float(steering)
