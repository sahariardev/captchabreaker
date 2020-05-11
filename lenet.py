from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential


class LeNet:
    @staticmethod
    def build(width, height, dept, classes):
        input_shape = (height, width, dept)
        if K.image_data_format() == "channels_first":
            input_shape = (dept, height, width)
        model = Sequential([
            Conv2D(20, (5, 5), padding="same", activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(20, (5, 5), padding="same", activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(500,activation='relu'),
            Dense(classes,activation='softmax')
        ])
        return model
