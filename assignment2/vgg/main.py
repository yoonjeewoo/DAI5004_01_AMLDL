import os
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import datasets, optimizers
import numpy as np

from network import VGG16

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def prepare_cifar(x, y):

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y


def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def main():

    print('loading data...')
    (x,y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)
    print(x.shape, y.shape, x_test.shape, y_test.shape)

    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')

    model = VGG16()
    model.build(input_shape=(None, 32, 32, 3))

    model.summary()

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)


    for epoch in range(50):

        for step, (x, y) in enumerate(train_loader):

            y = tf.one_hot(tf.squeeze(y, axis=1), depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)    
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()


        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                y = tf.one_hot(tf.squeeze(y, axis=1), depth=10)

                logits = model.predict(x)

                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()

if __name__ == '__main__':
    main()