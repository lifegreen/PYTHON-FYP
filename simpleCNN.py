print("At least the script starts...")
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.utils import to_categorical

from scipy.io import loadmat
import os
import re

import matplotlib.pyplot as plt

from datetime import datetime

def isListEmpty(inList):
# Checks if a list consists only of Empty nested lists
        if isinstance(inList, list): # Is a list (returns false if empty)???
            return all(map(isListEmpty, inList))
        return False # Not a list

class NinaproDB:
    winSize = 25
    overlap = 0

    def __init__(self):
        ## PLZ PLZ PLZ FIX ##
        # Created nested lists to store database
        #                               gestures             reps                 subjects            exercises
        self.Data = [ [ [ [ [] for i in range(13) ] for i in range(11) ] for i in range(64)] for i in range(3)]
        return

    def __str__(self):
        gesCount = [0 for i in range(13)]
        subCount = 0

        for exe in self.Data:
            for sub in exe:
                if not isListEmpty(sub):
                    subCount += 1
                    for rep in sub:
                        for ges in range(len(rep)):
                            gesCount[ges] += len(rep[ges])

        string = "%d %s \n" % (subCount, 'subject' if subCount==1 else 'subjects')
        string += "%d windows total\n" % sum(gesCount)
        for ges in gesCount:
            string += "Gesture %d - %d windows\n" % (gesCount.index(ges), ges)
        return string

    def readDataBase(self, folder, subject, exercise):
        subjects  = r'\d' if subject == 'all' else str(subject).replace(', ', '')
        exercises = r'\d' if exercise == 'all' else str(exercise).replace(', ', '')
        match = r'\AS{}+_.*E{}.*\.mat\Z'.format(subjects, exercises)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if re.search(match, file):
                     self.read(os.path.join(root, file))

    def read(self, file):
        print('Reading: %s', file)
        dataset = loadmat(file)
        signal = dataset['emg']
        labels = dataset['restimulus']
        exe    = dataset['exercise']
        sub    = dataset['subject']
        rep    = dataset['rerepetition']

        # Divide signal into time windows
        # length = (len(signal) // winSize) * winSize

        step = self.winSize - self.overlap
        # self.windows = [signal[i : i + self.winSize] for i in range(0, len(signal), step)]
        # if (self.windows[-1].shape != self.windows[1].shape) :
        #     del self.windows[-1]

        for start in range(0, len(signal) - self.winSize, step):
            end = start + self.winSize
            if labels[start] == labels[end-1]:
#PLS FIX#
                self.Data[exe[0][0]-1][sub[0][0]-1][rep[start][0]-1][labels[start][0]].append(signal[start:end])
# [0] are to turn a for some reason nested list into a scalar
# -1  is for converting to 0 indexing

            # print(start)
            # print(end)
            # print(exe[0][0])
            # print(sub[0][0])
            # print(rep[start][0])
            # print(labels[start][0])
            # print()


    def prepareData(self, ratio):
        TrainX = []
        TrainY = []
        ValidX = []
        ValidY = []
        # Data = self.Data[:][:][:][1][:]
        count = 0
        for exe in self.Data:
            if not isListEmpty(exe):
                for sub in exe:
                    if not isListEmpty(sub):
                        for rep in sub:
                            for ges, num in zip(rep, range(len(rep))):
                                if not num == 0:
                                    length = len(ges)

                                    middle = int(length * ratio)

                                    TrainX.extend(ges[middle:]) # |middle_____:
                                    ValidX.extend(ges[:middle]) # :____ |middle
                                    TrainY.extend( [num] * (length - middle) )
                                    ValidY.extend( [num] * middle )

        return [TrainX, TrainY , ValidX, ValidY]

## IMPORT DATA ##


DB = NinaproDB()
DB.readDataBase(r'C:\Users\Mark\Downloads\Datbase 1', 'all', '1')


print(DB)

[TrainX, TrainY , ValidX, ValidY] = DB.prepareData(0.3)
print(list(map(len, [TrainX, TrainY , ValidX, ValidY])))

TrainX = [np.expand_dims(x, axis=2) for x in TrainX]
TrainX = np.stack(TrainX, axis=0 )

ValidX = [np.expand_dims(x, axis=2) for x in ValidX]
ValidX = np.stack(ValidX, axis=0 )

print(TrainX[0].shape)
print(set(ValidY))


## CREATE NETWORK ##
num_filters = 8
filter_size = 3
pool_size = 2
num_classes = 13

model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=1,
#                  activation='relu',
#                  input_shape=(25, 10, 1), data_format="channels_last"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))

# model.add(Dense(num_classes, activation='softmax'))

## PAPER 2 ##
# model.add(Conv2D(32, kernel_size=(1,10), activation='relu', input_shape=(25,10,1), padding='same'))
# model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
# # model.add(AveragePooling2D(pool_size=(3,3)))
# model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
# # model.add(AveragePooling2D(pool_size=(3,3)))
# model.add(Conv2D(64, kernel_size=(5,1), activation='relu', padding='same'))
# model.add(Conv2D(64, kernel_size=1, activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))

## PAPER 3 ##
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(25,10,1), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

print(model)

#compile model using accuracy to measure model performance
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## TRAIN THE MODEL.##
model.fit(
    TrainX,
    to_categorical(TrainY),
    epochs=30,
    batch_size=128,
    validation_data=(ValidX, to_categorical(ValidY)),
    shuffle=True
)


# Save/Load weights
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")
model.save_weights('%s.h5' % dt_string)


# Predict on the first 5 test images.
predictions = np.argmax(model.predict(ValidX), axis=1)

# Print our model's predictions.
print('Predictions:\t', predictions[:20]) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print('Answers:\t\t', ValidY[:20]) # [7, 2, 1, 0, 4]

correct = predictions[ValidY==predictions]

print('Unique (pred):\t\t', set(predictions))
print('Unique (correct):\t', set(correct))

# plt.subplot(122)
# plt.hist(predictions, bins=11)  # arguments are passed to np.histogram
# print(np.histogram(predictions, bins=11))
# plt.title('Predictions')

# plt.subplot(121)
# plt.hist(correct, bins=11)  # arguments are passed to np.histogram
# plt.title('Correct')
# plt.show()