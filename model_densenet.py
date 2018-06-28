import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv
import cv2
import numpy as np
from models.densenet import DenseNet


def main():
    # read csv files for driving log
    lines = []
    with open('./data/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    correction = 0.2  # This is a parameter to tune
    current_path = './data/data/'
    # modify the image path
    images = []
    measurements = []
    for line in lines[1:]:
        img_centre = line[0]
        img_left = line[1][1:]  # get rid of the space at the beginning
        img_right = line[2][1:]  # get rid of the space at the beginning
        # create adjusted steering meaurements for the side camera iamges
        steering_centre = float(line[3])

        for num, img in enumerate([img_centre, img_left, img_right]):
            image = cv2.imread(current_path + img)
            images.append(image)
            images.append(cv2.flip(image, 1))

            if num == 1:
                steering = steering_centre + correction
            elif num == 2:
                steering = steering_centre - correction
            elif num == 0:
                steering = steering_centre

            measurements.append(steering)
            measurements.append(steering * (-1.0))

    X_train = np.array(images)
    y_train = np.array(measurements)

    # Load the model
    model = DenseNet(input_shape=(160, 320, 3), depth=3*3+4, nb_dense_block=3, growth_rate=12, bottleneck=True,
                     reduction=0.5, weights=None,
                     cropping=((70, 25), (0, 0)), norm_lambda=True,
                     activation=None, classes=1)

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    model.save('./models/model_densenet.h5')


if __name__ == '__main__':
    main()