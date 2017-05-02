from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

DATA_PATH = "training_data/data"
LABEL_PATH = os.path.join(DATA_PATH, "driving_log.csv")
BATCH_SIZE = 65
EPOCHS = 55

def keras_model(input_image):
    model = Sequential()

    model.add(Lambda(resize_images, input_image=input_image))
    model.add(Lambda(lambda x: x / 255. - 0.5))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model

def re_size_images(img):
# Keras needs the resizing.
return tf.image.resize_images(img, (66, 200))


def drop_low_steering_data_points(data):
# Drop data having low steering angles 
    idx = data[abs(data['steer']) < .05].index.tolist()
    dropped_rows = [j for j in idx if np.random.randint(10) < 8]
    saved_data = data.drop(data.index[dropped_rows])
    print("Dropped %s rows with low steering" % (len(dropped_rows)))
return saved_data


def get_random_image_and_steer_angle(data, value, data_path):
# A randomly selected right, left or center image and its steer angle is returned.
# The probability of selecting center image is twice of right or left image. 
    angle = .25
    random = np.random.randint(4)
    if random == 1 or random == 3:
        image_path = data['center'][value].strip()
        shift_ang = 0.0
    if random == 2:
        image_path = data['right'][value].strip()
        shift_ang = - angle
    if random == 0:
        image_path = data['left'][value].strip()
        shift_ang = angle
    image = get_cropped_image(plt.imread(os.path.join(data_path, image_path)))
    steer_angle = float(data['steer'][value]) + shift_ang
return image, steer_angle

def translated_image(image, steer):
# Returns translated image and its corresponding steer angle.
    translated_range = 100
    translated_x_axis = translated_range * np.random.uniform() - translated_range / 2
    steer_angle = steer + translated_x_axis / translated_range * 2 * .2
    translated_y_axis = 0
    translation = np.float32([[1, 0, translated_x_axis], [0, 1, translated_y_axis]])
    translated_image = cv2.warpAffine(image, translation, (320, 75))
return translated_image, steer_angle


def get_cropped_image(img):
# Returns cropped image
return img[60:135, :]

def generate_training_image(image_data, batch_size, data_path):
# Training data generator
    while True:
        # Fetch randomly sampled data from given pandas df
        batch = image_data.sample(n=batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels = np.empty([batch_size, 1])
        for idx, val in enumerate(batch.index.values):
            # Randomly select right, center or left image
            selected_image, steer_angle = get_random_image_and_steer_angle(image_data, val, data_path)
            selected_image = selected_image.reshape(selected_image.shape[0], selected_image.shape[1], 3)
            # Random Translation Jitter
            selected_image, steer_angle = translated_image(selected_image, steer_angle)

            # Flip Image randomly
            random = np.random.randint(1)
            if random == 0:
                selected_image, steer_angle = np.fliplr(selected_image), -steer_angle
            features[idx] = selected_image
            labels[idx] = steer_angle
            yield np.array(features), np.array(labels)


def get_images(image_data, data_path):
 # Validation Generator
    while True:
        for idx in range(len(image_data)):
            image_path = image_data['center'][idx].strip()
            image = get_cropped_image(plt.imread(os.path.join(data_path, image_path)))
            image = image.reshape(1, image.shape[0], image.shape[1], 3)
            steer_angle = image_data['steer'][idx]
            steer_angle = np.array([[steer_angle]])
            yield image, steer_angle


# Load CSV 
csv_file = pd.read_csv(LABEL_PATH, index_col=False)
csv_file.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']

# Shuffle data
csv_file = csv_file.sample(n=len(csv_file))

# Split data into Training data (80%) and Validation Data (20%)
training_size = int(0.8 * len(csv_file))
training_data = csv_file[:training_size].reset_index()
validation_data = csv_file[training_size:].reset_index()

# Drop low steering angles data to remove bias
training_data = drop_low_steering_data_points(training_data)

# Get input size for model by opening a random image
image = get_cropped_image(plt.imread(os.path.join(DATA_PATH, training_data['center'].iloc[909].strip())))

# Creating Model
model = keras_model(image.shape)
size_per_epoch = int(len(training_data) / BATCH_SIZE) * BATCH_SIZE
nb_val_samples = len(validation_data)

values = model.fit_generator(generate_training_image(training_data, BATCH_SIZE, DATA_PATH),
                             samples_per_epoch=size_per_epoch, nb_epoch=EPOCHS,
                             validation_data=get_images(validation_data, DATA_PATH),
                             nb_val_samples=nb_val_samples)

# Save model
model.save('model.h5')
