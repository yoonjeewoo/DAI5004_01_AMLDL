import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers, models
from    tensorflow.keras import regularizers

class VGG16(models.Model):

    def __init__(self):
        """
        :param input_shape: [32, 32, 3]
        """
        super(VGG16, self).__init__()

        self.num_classes = 10
        self.conv_block1 = keras.Sequential([
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.conv_block2 = keras.Sequential([
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.conv_block3 = keras.Sequential([
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.conv_block4 = keras.Sequential([
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.conv_block5 = keras.Sequential([
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(512)
        self.classification = layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classification(x)
        return x