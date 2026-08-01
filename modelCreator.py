import numpy as np
import cv2
import csv
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import Sequential, layers, optimizers
import matplotlib.pyplot as plt
from keras.models import save_model, load_model
import os

# Added for data augmentation, shared preprocessing, and disk-backed batching
from batch_generator import batch_generator, steps_per_epoch

# DATA
image_list = []
wheel_list = []

# Default 
DATA_DIR = os.environ.get('CVI620_DATA_DIR', r'C:\CVI620NSATestingData')
CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')

with open(CSV_PATH, mode="r", encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    for index, row in enumerate(csv_reader, start=1):

        print(f"Row {index}: {row}")

        # Switched from flattened 32×32 images to simulator compatible preprocessing and batching, so on the same 200×66 image pipeline used at test time
        # Remapped CSV paths to the local IMG/ folder
        center_path = row[0].strip()
        filename = os.path.basename(center_path.replace('\\', '/'))
        local_path = os.path.join(DATA_DIR, 'IMG', filename)
        image_list.append(local_path)
        wheel_list.append(row[3])

        if index%200==0:

            print(f'[INFO] {index} images processed!')

# Paths for the generator
X = np.array(image_list)
y = np.array(wheel_list, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replaced the flattened dense network with a CNN trained on 200×66 image tensors with the old model collapse
model = Sequential([
    layers.Lambda(lambda x: x - 0.5, input_shape=(66, 200, 3)),
    layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
    layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
    layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
    layers.Conv2D(64, (3, 3), activation='elu'),
    layers.Conv2D(64, (3, 3), activation='elu'),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(100, activation='elu'),
    layers.Dense(50, activation='elu'),
    layers.Dense(10, activation='elu'),
    layers.Dense(1, activation='linear'),
])

# Lowered the learning rate to make optimization more stable, fixed the model collapse issue
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='mean_squared_error',
              metrics=['mae'])

BATCH_SIZE = 32
# Increased the epoch count for better performance
EPOCHS = 60
train_gen = batch_generator(X_train, y_train, batch_size=BATCH_SIZE, training=True, aug_prob=0.5)
val_gen = batch_generator(X_test, y_test, batch_size=BATCH_SIZE, training=False)
train_steps = steps_per_epoch(len(X_train), BATCH_SIZE)
val_steps = steps_per_epoch(len(X_test), BATCH_SIZE)

if hasattr(model, 'fit_generator'):
    # Older Keras/TF from package_list.txt used fit_generator
    H = model.fit_generator(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        verbose=1,
    )
else:
    H = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        verbose=1,
    )

# EVALUATION
plt.figure(figsize=(10, 5))
plt.hist(y, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Steering Wheel Angles', fontsize=14)
plt.xlabel('Steering Angle (Radians/Degrees)', fontsize=12)
plt.ylabel('Number of Images (Frequency)', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig("steering_histogram.png", dpi=200)
print("Histogram saved.")

save_model(model, 'baseSelfDrivingCarModel.h5')
save_model(model, 'model.h5')
