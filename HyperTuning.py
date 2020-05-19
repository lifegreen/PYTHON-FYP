import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import filtfilt, butter

from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1, l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


class NinaproDB:
    fs = 100
    fc = 1  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = butter(5, w, 'low')

    winSize = int(0.256 * fs)
    step = int(0.032 * fs)

    subjects  = []
    exercises = []
    gestures  = []

    def __init__(self):
        print('winSize:', self.winSize)
        print('step:', self.step)
        self.Data = {'sub':[], 'exe':[], 'ges':[], 'rep':[], 'win':[]}

    def __repr__(self):
        subCount = len(self.subjects)
        exeCount = len(self.exercises)
        gesCount = len(self.gestures)
        winCount = len(self.Data['win'])

        string = self.__class__.__name__ + " with:\n"
        string += "%d %s \n" % (subCount, 'subject' if subCount==1 else 'subjects')
        string += "%d %s \n" % (exeCount, 'excercise' if exeCount==1 else 'excercises')
        string += "%d %s \n" % (gesCount, 'gesture' if gesCount==1 else 'gestures')
        string += "%d %s \n" % (winCount, 'window' if gesCount==1 else 'windows')

        for ges in self.gestures:
            string += "Gesture %d - %d windows\n" % (ges, self.Data['ges'].count(ges))

        return string

    def readDataBase(self, folder, subject, exercise):
        subjects  = r'\d+' if (subject == 'all') else str(subject).replace(', ', '')
        exercises = r'\d' if (exercise == 'all') else str(exercise).replace(', ', '')

        match = r'\AS{}_.*E{}.*\.mat\Z'.format(subjects, exercises)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if re.search(match, file):
                    self.read(os.path.join(root, file))

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])


    def read(self, file):
        print('Reading:', file)
        dataset = loadmat(file)
        signal = dataset['emg']

#         for c in range(signal.shape[1]):
#             signal[:,c] = filtfilt(self.b, self.a, signal[:,c])
#             X = signal
#             X[:,c] /= max(abs(X[:,c]))
#             X[:,c] += 10*c

#         plt.figure()
#         plt.plot(X)

        # By default this returns list of lists, each with a single scalar.
        # So np.squeeze is used to turn it into a vector.
        ges = np.squeeze(dataset['restimulus'])
        rep = np.squeeze(dataset['rerepetition'])

        # item() is used to return get scalars instead of lists
        exe = dataset['exercise'].item()
        sub = dataset['subject'].item()

        gestStart = next((i for i, x in enumerate(ges) if x!=0)) # returns index of the frist nonzero element

        # Find where the first rest period ends
        restEnd = next((i for i, x in enumerate(ges) if x==2))

        for start in range(gestStart, len(signal) - self.winSize, self.step):
            end = start + self.winSize

            # Make sure the same gesture is performed throught the window duaration
            if (ges[start] != 0) or (end < restEnd):
                if ges[start] == ges[end-1]:
                    self.Data['sub'].append(sub)
                    self.Data['exe'].append(exe)
                    self.Data['rep'].append(rep[start])
                    self.Data['ges'].append(ges[start])
                    self.Data['win'].append(signal[start:end])

    def readMat(self, folder):

        for i in range(9):
            file = 'motion' + str(i) + '.mat'

            print('Reading:', file)
            motion = loadmat(os.path.join(folder, file))['motion' + str(i)]
            motion = motion[~np.all(motion == 0, axis=1)] # remove rows with all zero values

#             for c in range(motion.shape[1]):
#                 motion[:,c] = filtfilt(self.b, self.a, motion[:,c])

            for start in range(0, len(motion) - self.winSize, self.step):
                end = start + self.winSize

                self.Data['ges'].append(i)
                self.Data['win'].append(motion[start:end])

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])

    def splitByRatio(self, ratio):
        [TrainX, ValidX, TrainY, ValidY] = train_test_split(self.Data['win'], self.Data['ges'], test_size=ratio, shuffle=True)
        return [TrainX, TrainY, ValidX, ValidY]

    def splitByCategory(self, name, ls):
        DF = pd.DataFrame(self.Data)

        TrainX = []
        TrainY = []
        ValidX = []
        ValidY = []

        validIdxs = False * len(self.Data)
        for i in ls:
            validIdxs |= (DF[name] == i)

        ValidX = list(DF['win'][validIdxs])
        ValidY = list(DF['ges'][validIdxs])

        TrainX = list(DF['win'][~validIdxs])
        TrainY = list(DF['ges'][~validIdxs])

        return [TrainX, TrainY, ValidX, ValidY]


class NinaproDB:
    fs = 100
    fc = 1  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = butter(5, w, 'low')

    winSize = int(0.256 * fs)
    step = int(0.032 * fs)

    subjects  = []
    exercises = []
    gestures  = []

    def __init__(self):
        print('winSize:', self.winSize)
        print('step:', self.step)
        self.Data = {'sub':[], 'exe':[], 'ges':[], 'rep':[], 'win':[]}

    def __repr__(self):
        subCount = len(self.subjects)
        exeCount = len(self.exercises)
        gesCount = len(self.gestures)
        winCount = len(self.Data['win'])

        string = self.__class__.__name__ + " with:\n"
        string += "%d %s \n" % (subCount, 'subject' if subCount==1 else 'subjects')
        string += "%d %s \n" % (exeCount, 'excercise' if exeCount==1 else 'excercises')
        string += "%d %s \n" % (gesCount, 'gesture' if gesCount==1 else 'gestures')
        string += "%d %s \n" % (winCount, 'window' if gesCount==1 else 'windows')

        for ges in self.gestures:
            string += "Gesture %d - %d windows\n" % (ges, self.Data['ges'].count(ges))

        return string

    def readDataBase(self, folder, subject, exercise):
        subjects  = r'\d+' if (subject == 'all') else str(subject).replace(', ', '')
        exercises = r'\d' if (exercise == 'all') else str(exercise).replace(', ', '')

        match = r'\AS{}_.*E{}.*\.mat\Z'.format(subjects, exercises)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if re.search(match, file):
                    self.read(os.path.join(root, file))

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])


    def read(self, file):
        print('Reading:', file)
        dataset = loadmat(file)
        signal = dataset['emg']

#         for c in range(signal.shape[1]):
#             signal[:,c] = filtfilt(self.b, self.a, signal[:,c])
#             X = signal
#             X[:,c] /= max(abs(X[:,c]))
#             X[:,c] += 10*c

#         plt.figure()
#         plt.plot(X)

        # By default this returns list of lists, each with a single scalar.
        # So np.squeeze is used to turn it into a vector.
        ges = np.squeeze(dataset['restimulus'])
        rep = np.squeeze(dataset['rerepetition'])

        # item() is used to return get scalars instead of lists
        exe = dataset['exercise'].item()
        sub = dataset['subject'].item()

        gestStart = next((i for i, x in enumerate(ges) if x!=0)) # returns index of the frist nonzero element

        # Find where the first rest period ends
        restEnd = next((i for i, x in enumerate(ges) if x==2))

        for start in range(gestStart, len(signal) - self.winSize, self.step):
            end = start + self.winSize

            # Make sure the same gesture is performed throught the window duaration
            if (ges[start] != 0) or (end < restEnd):
                if ges[start] == ges[end-1]:
                    self.Data['sub'].append(sub)
                    self.Data['exe'].append(exe)
                    self.Data['rep'].append(rep[start])
                    self.Data['ges'].append(ges[start])
                    self.Data['win'].append(signal[start:end])

    def readMat(self, folder):

        for i in range(9):
            file = 'motion' + str(i) + '.mat'

            print('Reading:', file)
            motion = loadmat(os.path.join(folder, file))['motion' + str(i)]
            motion = motion[~np.all(motion == 0, axis=1)] # remove rows with all zero values

#             for c in range(motion.shape[1]):
#                 motion[:,c] = filtfilt(self.b, self.a, motion[:,c])

            for start in range(0, len(motion) - self.winSize, self.step):
                end = start + self.winSize

                self.Data['ges'].append(i)
                self.Data['win'].append(motion[start:end])

        self.subjects  = set(self.Data['sub'])
        self.exercises = set(self.Data['exe'])
        self.gestures  = set(self.Data['ges'])

    def splitByRatio(self, ratio):
        [TrainX, ValidX, TrainY, ValidY] = train_test_split(self.Data['win'], self.Data['ges'], test_size=ratio, shuffle=True)
        return [TrainX, TrainY, ValidX, ValidY]

    def splitByCategory(self, name, ls):
        DF = pd.DataFrame(self.Data)

        TrainX = []
        TrainY = []
        ValidX = []
        ValidY = []

        validIdxs = False * len(self.Data)
        for i in ls:
            validIdxs |= (DF[name] == i)

        ValidX = list(DF['win'][validIdxs])
        ValidY = list(DF['ges'][validIdxs])

        TrainX = list(DF['win'][~validIdxs])
        TrainY = list(DF['ges'][~validIdxs])

        return [TrainX, TrainY, ValidX, ValidY]

## IMPORT DATA ##

DB = NinaproDB()
DB.readDataBase(r'C:\Users\Mark\Downloads\Database 1', [1,7], 1)

print(DB)


# [TrainX, TrainY, ValidX, ValidY] = DB.splitByCategory('rep', [2, 5, 7])
[TrainX, TrainY, ValidX, ValidY] = DB.splitByRatio(0.33)
lengths = list(map(len, [TrainX, TrainY, ValidX, ValidY]))
print(lengths)
print('Ratio:', lengths[3]/(lengths[3]+lengths[1]))

# Add a colour channel dimension for conv net input
TrainX = [np.expand_dims(x, axis=2) for x in TrainX]
TrainX = np.stack(TrainX, axis=0 )

ValidX = [np.expand_dims(x, axis=2) for x in ValidX]
ValidX = np.stack(ValidX, axis=0 )

print(TrainX[0].shape)
print(set(TrainY))
print(set(ValidY))


from kerastuner import HyperModel


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()

        decay_rate = hp.Choice('Decay', values=[0.0, 0.0001, 0.01]) #0.0005, 0.001,

#         if hp.Boolean('Decay_type'):
#             decay = l1(decay_rate)
#         else:
#             decay = l2(decay_rate)
        decay = l2(decay_rate)

        Norm = hp.Boolean('Batch_Norm')
        Drop = hp.Boolean('Dropout')

        #Conv1
        model.add(Conv2D(
            filters=64,

            kernel_size=hp.Choice('Conv1_size', values=[3, 5]),

            activation='relu', padding='same', kernel_regularizer=decay, input_shape=self.input_shape
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        if Norm:
            model.add(BatchNormalization())

        #Conv2
        model.add(Conv2D(
            filters=hp.Choice('Conv2_filtN', values=[32,64]),

            kernel_size=hp.Choice('Conv2_size', values=[3, 5]),

            activation='relu', padding='same', kernel_regularizer=decay
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        if Norm:
            model.add(BatchNormalization())

        if Drop:
            model.add(Dropout(.2))

        #Conv3
        if hp.Boolean('Conv3_layer'):
            model.add(Conv2D(32, kernel_size=3, activation='relu',  padding='same', kernel_regularizer=decay))
            if Norm:
                model.add(BatchNormalization())

        # Dense
        model.add(Flatten())

#         if hp.Boolean('Extra_Dense_layer'):
#             model.add(Dense(1000, activation='relu'))
        if Drop:
            model.add(Dropout(.4))

        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=decay))

        # optimizer
        learning_rate = hp.Choice('Learning_rate', values=[0.01, 0.001], default=0.001)

        optimizer = hp.Choice('Optimizer', values=['a', 'b', 'c'], default = 'a')

        if optimizer == 'a':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'b':
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


from kerastuner import Hyperband

NUM_CLASSES = len(DB.gestures)
INPUT_SHAPE = TrainX[0].shape

HYPERBAND_MAX_EPOCHS = 30
EXECUTION_PER_TRIAL  = 1

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

tuner = Hyperband(
    hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_acc',
#     seed=SEED,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='hyperband',
    project_name='test_1'
)

tuner.search(TrainX, to_categorical(TrainY), batch_size=64, validation_data=(ValidX, to_categorical(ValidY)), verbose=0)

tuner.results_summary()