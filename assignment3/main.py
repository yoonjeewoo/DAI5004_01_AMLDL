import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets, optimizers
from tensorflow.python.keras.layers.recurrent import LSTM
from network import LSTM_FC
from tqdm import tqdm

def main():
    max_features = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review

    print('loading data...')
    (x_train, y_train), (x_val, y_val) = datasets.imdb.load_data(
        num_words=max_features
    )

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_loader = train_loader.shuffle(25000).batch(32)

    test_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader = test_loader.shuffle(25000).batch(32)
    print('done.')
    

    print(x_train.shape)

    model = LSTM_FC(num_classes=2, max_features=max_features)
    model.build(input_shape=(None, 200))

    model.summary()

    loss_fn = keras.losses.CategoricalCrossentropy()
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    for epoch in range(50):

        for step, (x, y) in enumerate(tqdm(train_loader)):
            y = tf.one_hot(y, depth=2)
            # print(y)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)    
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # if step % 100 == 0:
            tqdm.write(f"EPOCH: {epoch+1} STEP: {step} loss: {float(loss):.2f} acc: {metric.result().numpy():.2f}")
        metric.reset_states()


        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in tqdm(test_loader):
                y = tf.one_hot(y, depth=2)

                logits = model.predict(x)

                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()


if __name__ == '__main__':
    main()
    