# dog vs. cat classifier with transferred learning from r22 (fashion mnist)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from time import time
import matplotlib.pyplot as plt

from tfutils import MyCallbackStopTraining

N_EPOCHS = 10
MODEL = 'cnn_tl'
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

# transfer learning parameters:
MODEL_NAME_ORIGINAL = 'cnn_1616132484'
LAST_LAYER_IND = -3

TARGET_ACC_CALLBACK = .99
TARGET_ACC_SAVE = .7


# next: image augmentation, loop for hyper-parameter search
def main():
    # (1) create model:
    pre_trained_model = tf.keras.models.load_model('saved_models/' + MODEL_NAME_ORIGINAL)
    pre_trained_model.summary()
    pre_trained_model.load_weights('saved_weights/' + MODEL_NAME_ORIGINAL)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    pre_trained_model.summary()
    last_layer = pre_trained_model.get_layer(pre_trained_model.layers[LAST_LAYER_IND].name)
    x = tf.keras.layers.Dense(1000, activation='relu')(last_layer.output)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(pre_trained_model.input, x)
    model.summary()

    # (2) prepare data:
    train_dir = 'data/dog_vs_cat/dataset/training_set'
    test_dir = 'data/dog_vs_cat/dataset/test_set'
    target_size = model.get_layer(model.layers[0].name).input_shape[0][1:-1]
    data_gen = ImageDataGenerator(rescale=1 / 255.0)
    train_data_gen = data_gen.flow_from_directory(train_dir, target_size=target_size, batch_size=20)
    test_data_gen = data_gen.flow_from_directory(test_dir, target_size=target_size, batch_size=40)

    # (3) train model:
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    callback = MyCallbackStopTraining(target=TARGET_ACC_CALLBACK)
    start = time()
    history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=N_EPOCHS, callbacks=[callback], verbose=1)
    training_time = np.round(time() - start, 2)

    # (4) predict on test data:
    loss, acc = model.evaluate(test_data_gen)
    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing acc=' + str(np.round(acc, 5)) + '\n')

    # (5) save model, plots:
    if acc > TARGET_ACC_SAVE and SAVE_MODEL:
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
