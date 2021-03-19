import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

from models import m22_cnn_cifar10
from tfutils import model_checkpoint_callback, create_path_if_not_exists


N_EPOCHS = 5
MODEL = 'cnn'
MODEL_NAME = MODEL + '_' + str(int(time()))
PLOTS = True


# next: image augmentation, loop for hyper-parameter search
def main():
    # (1) prepare data:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # (2) train model:
    model = m22_cnn_cifar10.Model(MODEL, input_shape=(32, 32, 3)).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
    start = time()
    model.save('saved_models/' + MODEL_NAME)
    weights_dir = create_path_if_not_exists('saved_weights/' + MODEL_NAME)
    callback = model_checkpoint_callback(weights_dir, monitor='val_acc', mode='max')
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=N_EPOCHS, verbose=1,
                        callbacks=[callback])
    training_time = np.round(time() - start, 2)

    # (3) predict on test data:
    loss, acc = model.evaluate(x_test, y_test)
    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing acc=' + str(np.round(acc, 5)) + '\n')

    # (4) save model, plots:
    model.save('saved_models/' + MODEL_NAME)
    print('saved model as {}'.format(MODEL_NAME))

    if PLOTS:
        plt.plot(history.epoch, history.history['loss'])
        plt.plot(history.epoch, history.history['val_loss'], color='orange')
        plt.title('training loss')
        plt.savefig('results/' + MODEL_NAME + '_loss_dvc.png')
        print('saved plot: ' + 'results/' + MODEL_NAME + '_loss_dvc.png')
    print('----------')


if __name__ == '__main__':
    main()
