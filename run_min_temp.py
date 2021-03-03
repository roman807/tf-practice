import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from time import time
from datetime import datetime

from tfutils import windowed_dataset, MyCallback
from models import rnn_min_temp


TRAIN_SPLIT = .75
N_EPOCHS = 10
WINDOW_SIZE = 5
MODEL_NAME = 'rnn_' + str(int(time()))


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
    model = rnn_min_temp.get_model()
    callbacks = MyCallback(metric='mae', greater=False, target=1.5)
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    history = model.fit(ds_train, validation_data=ds_test, epochs=N_EPOCHS, callbacks=[callbacks], verbose=1)

    # predict on test data:
    y_pred = model.predict(ds_test).flatten()
    dates_test = dates[split+WINDOW_SIZE:]
    y_test = temps[split+WINDOW_SIZE:]
    mae = np.round(tf.keras.metrics.mae(y_test, y_pred).numpy(), 5)

    # plot training
    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'], color='orange')
    plt.title('training loss')
    plt.savefig('results/' + MODEL_NAME + '_loss.png')
    # plt.show()

    # plot results
    plt.plot(dates_test, y_test)
    plt.plot(dates_test, y_pred, color='orange')
    plt.title('mean absolute error: {}'.format(mae))
    plt.savefig('results/' + MODEL_NAME + '_results.png')
    # plt.show()
    # a=1


if __name__ == '__main__':
    main()
