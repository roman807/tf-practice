# download dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud
# upload to remote with scp archive.zip dsvm-user@13.69.252.2:/tmp/pycharm_project_982/data
# unzip archive.zip

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score

from models import m10_dnn_credit
from tfutils import model_checkpoint_callback

TRAIN_SPLIT = .75
N_EPOCHS = 7
MODEL = 'dnn'
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

# TARGET_ACC_CALLBACK = .99
TARGET_F1_SAVE = .7


def main():
    # (1) prepare data
    ds = pd.read_csv('data/creditcard.csv').drop(['Time', 'Amount'], axis=1).to_numpy()
    split = int(TRAIN_SPLIT * ds.shape[0])
    x_train = ds[:split, :-1]
    x_test = ds[split:, :-1]
    y_train = ds[:split, -1]
    y_test = ds[split:, -1]

    # (2) train model
    model = m10_dnn_credit.Model(model_type=MODEL, input_shape=[x_train.shape[1]]).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    checkpoint_filepath = '/tmp/checkpoint_weights'
    callbacks = model_checkpoint_callback(checkpoint_filepath=checkpoint_filepath, monitor='val_loss', mode='min')
    model.compile(optimizer=opt, loss='binary_crossentropy')
    start = time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=N_EPOCHS, callbacks=[callbacks],
                        batch_size=50, verbose=1)
    model.load_weights(checkpoint_filepath)
    training_time = time() - start

    # (3) predict on test data:
    loss = model.evaluate(x_test, y_test)
    y_pred = np.round(model.predict(x_test))
    f1 = f1_score(y_test, y_pred)

    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing f1-score=' + str(np.round(f1, 5)) + '\n')
    print('testing loss=' + str(np.round(loss, 6)) + '\n')

    # save model
    if f1 > TARGET_F1_SAVE and SAVE_MODEL:
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
