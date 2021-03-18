# to download data to remote machine with kaggle api:
# (1) Log in to machine: ssh [user]@[IP-address]
# (2) Activate env, e.g. conda activate /anaconda/envs/py37_tensorflow
# (3) pip install kaggle
# (4) Go to kaggle.com -> account -> create token
# (5) (log out of remote ->) scp kaggle.json dsvm-user@[IP-address]:/home/[user]/.kaggle
# (6) (log in to remote ->) chmod 600 /home/[user]/.kaggle/kaggle.json
# (7) kaggle datasets download -d chetankv/dogs-cats-images

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from time import time
import matplotlib.pyplot as plt

from models import m21_cnn_dog_vs_cat
from tfutils import MyCallbackStopTraining


N_EPOCHS = 60
MODEL = 'cnn2'
AUGMENT = True
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

TARGET_ACC_CALLBACK = .9
TARGET_ACC_SAVE = .7


# next: image augmentation, loop for hyper-parameter search
def main():
    # (1) prepare data:
    train_dir = 'data/dog_vs_cat/dataset/training_set'
    test_dir = 'data/dog_vs_cat/dataset/test_set'

    data_gen = ImageDataGenerator(rescale=1 / 255.0)
    if AUGMENT:
        data_gen_train = ImageDataGenerator(rescale=1 / 255.0, rotation_range=40, zoom_range=.2, shear_range=.2,
                                            width_shift_range=.2, height_shift_range=.2, vertical_flip=True)
        train_data_gen = data_gen_train.flow_from_directory(train_dir, target_size=(28, 28), batch_size=20)
    else:
        train_data_gen = data_gen.flow_from_directory(train_dir, target_size=(28, 28), batch_size=20)
    test_data_gen = data_gen.flow_from_directory(test_dir, target_size=(28, 28), batch_size=40)

    # (2) train model:
    model = m21_cnn_dog_vs_cat.Model(MODEL, input_shape=(28, 28, 3)).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1.0e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    callback = MyCallbackStopTraining(target=TARGET_ACC_CALLBACK)
    start = time()
    history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=N_EPOCHS, callbacks=[callback], verbose=1)
    training_time = np.round(time() - start, 2)

    # (3) predict on test data:
    loss, acc = model.evaluate(test_data_gen)
    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing acc=' + str(np.round(acc, 5)) + '\n')

    # (4) save model, plots:
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
