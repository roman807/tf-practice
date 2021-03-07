import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from time import time
from datetime import datetime

from tfutils import windowed_dataset, MyCallbackStopTraining, model_checkpoint_callback
from models import m40_rnn_min_temp


TRAIN_SPLIT = .75
N_EPOCHS = 200
WINDOW_SIZE = 10
MODEL = 'cnn'
MODEL_NAME = MODEL + '_' + str(int(time()))
SAVE_MODEL = True
PLOTS = True

TARGET_MAE_CALLBACK = 0.0   # set to 0.0 to prevent training stop
TARGET_MAE_SAVE = 3


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
    checkpoint_filepath = '/tmp/checkpoint_weights'
    callbacks = [
        MyCallbackStopTraining(metric='mae', greater=False, target=TARGET_MAE_CALLBACK),
        model_checkpoint_callback(checkpoint_filepath=checkpoint_filepath, monitor='val_loss', mode='min')]
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    start = time()
    history = model.fit(ds_train, validation_data=ds_test, epochs=N_EPOCHS, callbacks=callbacks, verbose=1)
    training_time = np.round(time() - start, 2)
    model.load_weights(checkpoint_filepath)

    # predict on test data:
    y_pred = model.predict(ds_test).flatten()
    dates_test = dates[split+WINDOW_SIZE:]
    y_test = temps[split+WINDOW_SIZE:]
    mae = np.round(tf.keras.metrics.mae(y_test, y_pred).numpy(), 5)
    print('----------')
    print('trained ' + MODEL + ' for ' + str(N_EPOCHS) + ' epochs in ' + str(training_time) + 'seconds')
    print('testing mae=' + str(np.round(mae, 5)) + '\n')

    # save model
    if mae < TARGET_MAE_SAVE and SAVE_MODEL:
        model.save('saved_models/' + MODEL_NAME)
        print('saved model as {}'.format(MODEL_NAME))

    if PLOTS:
        # plot training
        plot_start = 30
        plt.plot(history.epoch[plot_start:], history.history['loss'][plot_start:])
        plt.plot(history.epoch[plot_start:], history.history['val_loss'][plot_start:], color='orange')
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
