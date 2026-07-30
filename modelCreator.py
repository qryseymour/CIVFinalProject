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
            break

X = np.array(image_list)
y = np.array(wheel_list)
X_train, X_test, y_train, y_test = train_test_split(X, y)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# MODEL
model = Sequential([
    layers.Dense(20, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256)

# # EVALUATION
# plt.plot(np.arange(10), H.history['accuracy'], label='train accuracy')
# plt.plot(np.arange(10), H.history['val_accuracy'], label='test accuracy')
plt.plot(np.arange(10), H.history['loss'], label='train loss')
plt.plot(np.arange(10), H.history['val_loss'], label='test loss')
plt.plot(np.arange(10), H.history['accuracy'], label='train accuracy')
plt.plot(np.arange(10), H.history['val_accuracy'], label='test accuracy')
plt.legend()
plt.show()

save_model(model, 'baseSelfDrivingCarModel.h5')