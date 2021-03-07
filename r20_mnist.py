import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

from models import m20_cnn_mnist
from tfutils import MyCallbackStopTraining


N_EPOCHS = 5
MODEL = 'cnn'
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

TARGET_ACC_CALLBACK = .99
TARGET_ACC_SAVE = .95


def main():
    # (1) prepare data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # (2) train model
    model = m20_cnn_mnist.Model(model_type='cnn', input_shape=(28, 28)).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    callbacks = MyCallbackStopTraining(metric='accuracy', greater=True, target=TARGET_ACC_CALLBACK)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=N_EPOCHS, callbacks=[callbacks],
                        batch_size=50, verbose=1)
    training_time = np.round(time() - start, 2)

    # (3) predict on test data:
    # y_pred = np.argmax(model.predict(x_test), axis=1)
    # acc = tf.keras.metrics.Accuracy()(y_test, y_pred).numpy()
    loss, acc = model.evaluate(x_test, y_test)

    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing acc=' + str(np.round(acc, 5)) + '\n')

    # save model
    if acc > TARGET_ACC_SAVE and SAVE_MODEL:
        model.save('saved_models/' + MODEL_NAME)
        print('saved model as {}'.format(MODEL_NAME))

    if PLOTS:
        plt.plot(history.epoch, history.history['loss'])
        plt.plot(history.epoch, history.history['val_loss'], color='orange')
        plt.title('training loss')
        plt.savefig('results/' + MODEL_NAME + '_loss.png')
        print('saved plot: ' + 'results/' + MODEL_NAME + '_loss.png')
    print('----------')


if __name__ == '__main__':
    main()
