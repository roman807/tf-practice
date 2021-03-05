import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from time import time
from datetime import datetime

from tfutils import windowed_dataset, MyCallback
from models import m40_rnn_min_temp


TRAIN_SPLIT = .75
N_EPOCHS = 25
WINDOW_SIZE = 5
MODEL = 'rnn'
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

METRIC = 'mae'
TARGET_CALLBACK = 1.5
TARGET_SAVE = 3


def main():
    # (1) prepare data
    dates, temps = [], []
    with open('data/min_temp.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in f:
            row = row.replace('\n', '').replace('"', '').split(',')
            dates.append(datetime.strptime(row[0], '%Y-%m-%d').date())
            temps.append(float(row[1]))

    split = int(TRAIN_SPLIT * len(temps))
    ds_train = windowed_dataset(temps[:split], window_size=WINDOW_SIZE, batch_size=20, shuffle_buffer=100)
    ds_test = windowed_dataset(temps[split:], window_size=WINDOW_SIZE, batch_size=20, shuffle=False)

    # (2) train model
    model = m40_rnn_min_temp.Model(MODEL, WINDOW_SIZE).get_model()
    callbacks = MyCallback(metric='mae', greater=False, target=1.5)
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    start = time()
    history = model.fit(ds_train, validation_data=ds_test, epochs=N_EPOCHS, callbacks=[callbacks], verbose=1)
    training_time = np.round(time() - start, 2)

    # predict on test data:
    y_pred = model.predict(ds_test).flatten()
    dates_test = dates[split+WINDOW_SIZE:]
    y_test = temps[split+WINDOW_SIZE:]
    mae = np.round(tf.keras.metrics.mae(y_test, y_pred).numpy(), 5)
    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing mae=' + str(np.round(mae, 5)) + '\n')

    # save model
    if mae < TARGET_SAVE and SAVE_MODEL:
        model.save('saved_models/' + MODEL_NAME)
        print('saved model as {}'.format(MODEL_NAME))

    if PLOTS:
        # plot training
        plt.plot(history.epoch, history.history['loss'])
        plt.plot(history.epoch, history.history['val_loss'], color='orange')
        plt.title('training loss')
        plt.savefig('results/' + MODEL_NAME + '_loss.png')
        print('saved plot: ' + 'results/' + MODEL_NAME + '_loss.png')

        # plot results
        plt.plot(dates_test, y_test)
        plt.plot(dates_test, y_pred, color='orange')
        plt.title('mean absolute error: {}'.format(mae))
        plt.savefig('results/' + MODEL_NAME + '_results.png')
        print('saved plot: ' + 'results/' + MODEL_NAME + '_results.png')
    print('----------')


if __name__ == '__main__':
    main()
