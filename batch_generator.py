import numpy as np

from data_augmentation import augment_image
from preprocessing import load_rgb_image, preprocess_image


def batch_generator(image_paths, steering_angles, batch_size=32, training=True, aug_prob=0.5):
    """
    Generate batches of images and steering labels for model training or evaluation.
    
    - training=True: randomly augment a portion of samples, then preprocess
    - training=False: preprocess only
    """
    image_paths = np.asarray(image_paths)
    steering_angles = np.asarray(steering_angles, dtype=np.float32)
    n = len(image_paths)
    if n == 0:
        raise ValueError("batch_generator received an empty dataset")

    indices = np.arange(n)

    while True:
        if training:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            # Skip the final partial batch so batch dimensions stay consistent
            if len(batch_idx) < batch_size:
                continue

            X = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
            y = np.zeros(batch_size, dtype=np.float32)
            filled = 0

            # Load each image in the batch, optionally augment it then preprocess and store it
            for idx in batch_idx:
                img = load_rgb_image(image_paths[idx])
                if img is None:
                    continue

                angle = float(steering_angles[idx])
                if training:
                    img, angle = augment_image(img, angle, aug_prob=aug_prob)

                X[filled] = preprocess_image(img)
                y[filled] = angle
                filled += 1

                if filled == batch_size:
                    break

            # If some reads failed, top up from random samples
            while filled < batch_size:
                idx = np.random.randint(0, n)
                img = load_rgb_image(image_paths[idx])
                if img is None:
                    continue
                angle = float(steering_angles[idx])
                if training:
                    img, angle = augment_image(img, angle, aug_prob=aug_prob)
                X[filled] = preprocess_image(img)
                y[filled] = angle
                filled += 1

            yield X, y


def steps_per_epoch(num_samples, batch_size):
    """Return the number of full batches per epoch, with a minimum of 1"""
    return max(1, num_samples // batch_size)
    