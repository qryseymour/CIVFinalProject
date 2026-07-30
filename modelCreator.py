import numpy as np
import cv2
import csv
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import Sequential, layers
import matplotlib.pyplot as plt
from keras.models import save_model, load_model

# DATA
image_list = []
wheel_list = []

with open(r"C:\CVI620NSATestingData\driving_log.csv", mode="r", encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    for index, row in enumerate(csv_reader, start=1):

        print(f"Row {index}: {row}")

        image = cv2.imread(row[0])
        image = cv2.resize(image, (32, 32))
        image = image/255
        image = image.flatten()
        image_list.append(image)
        wheel_list.append(row[3])

        if index%200==0:

            print(f'[INFO] {index} images processed!')

X = np.array(image_list, dtype='float32')
y = np.array(wheel_list, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
model = Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# EVALUATION
plt.figure(figsize=(10, 5))
plt.hist(y, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Steering Wheel Angles', fontsize=14)
plt.xlabel('Steering Angle (Radians/Degrees)', fontsize=12)
plt.ylabel('Number of Images (Frequency)', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.show()

save_model(model, 'baseSelfDrivingCarModel.h5')