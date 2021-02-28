import tensorflow as tf
# import pandas as pd
import csv
from datetime import datetime

from tfutils import windowed_dataset
from models import rnn_min_temp


TRAIN_SPLIT = .75
N_EPOCHS = 2


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
    x_train = temps[:split]
    x_test = temps[split:]
    ds_train = windowed_dataset(x_train, window_size=5, batch_size=20, shuffle_buffer=100)
    ds_test = windowed_dataset(x_test, window_size=5, batch_size=20, shuffle=False)

    # (2) train model
    model = rnn_min_temp.get_model()

    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mse')
    model.fit(ds_train, epochs=N_EPOCHS, verbose=1)
    # model.fit(ds_train, validation_data=ds_test, epochs=N_EPOCHS, verbose=1)


if __name__ == '__main__':
    main()
