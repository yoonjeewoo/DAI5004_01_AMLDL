from tensorflow.keras import layers, models

class LSTM_FC(models.Model):

    def __init__(self, num_classes, max_features):
        super(LSTM_FC, self).__init__()

        self.num_classes = num_classes
        self.max_features = max_features
        self.embedding = layers.Embedding(self.max_features, 128)
        self.lstm = layers.LSTM(64)
        self.fc = layers.Dense(self.num_classes, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x