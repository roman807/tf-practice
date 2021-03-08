
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from time import time
from datetime import datetime
from collections import defaultdict

from tfutils import windowed_dataset, MyCallbackStopTraining, model_checkpoint_callback, clear_path
from models import m40_rnn_min_temp


TRAIN_SPLIT = .75
N_EPOCHS = 80
WINDOW_SIZE = 20

FIND_LR = False
LR_START = 1.0e-6

FIND_BEST_MODEL = True
TARGET_MAE_CALLBACK = 0.0   # set to 0.0 to prevent training stop
TARGET_MAE_SAVE = 3
COLORS = ['grey', 'blue', 'green', 'black', 'orange', 'red']

MODELS = {
    'cnn': 'cnn',
    # 'rnn': 'rnn',
    'cnn_rnn': 'cnn_rnn'
}
OPTIMIZERS = {
    # 'adam04': tf.keras.optimizers.Adam(lr=1e-4),
    'adam05': tf.keras.optimizers.Adam(lr=5e-5),
    # 'sgd05': tf.keras.optimizers.SGD(lr=5e-5)
}
# todo: next: try different window sizes
WINDOW_SIZES = {
    '5': 5,
    '10': 10,
    '20': 20
}


def get_data():
    dates, temps = [], []
    with open('data/min_temp.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in f:
            row = row.replace('\n', '').replace('"', '').split(',')
            dates.append(datetime.strptime(row[0], '%Y-%m-%d').date())
            temps.append(float(row[1]))
    split = int(TRAIN_SPLIT * len(temps))
    return (dates,
            temps,
            windowed_dataset(temps[:split], window_size=WINDOW_SIZE, batch_size=20, shuffle_buffer=100),
            windowed_dataset(temps[split:], window_size=WINDOW_SIZE, batch_size=20, shuffle=False))


def train_model(ds_train, ds_test, model, opt, ws, callbacks=[]):
    # model = m40_rnn_min_temp.Model(model, WINDOW_SIZE).get_model()
    model = m40_rnn_min_temp.Model(model, ws).get_model()
    checkpoint_filepath = clear_path('/tmp/checkpoint_weights')
    callbacks_ = callbacks.copy()
    callbacks_.append(MyCallbackStopTraining(metric='mae', greater=False, target=TARGET_MAE_CALLBACK))
    callbacks_.append(model_checkpoint_callback(checkpoint_filepath=checkpoint_filepath, monitor='val_loss', mode='min'))
    model.summary()
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    history = model.fit(ds_train, validation_data=ds_test, epochs=N_EPOCHS, callbacks=callbacks_, verbose=1)
    model.load_weights(checkpoint_filepath)
    return model, history


def main():
    tf.keras.backend.clear_session()

    # (1) prepare data
    dates, temps, ds_train, ds_test = get_data()
    split = int(TRAIN_SPLIT * len(temps))

    # (2a) find learning rate
    if FIND_LR:
        callbacks = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LR_START * 10**(epoch/20))
        # opt = tf.keras.optimizers.Adam(learning_rate=LR_START)
        opt = tf.keras.optimizers.SGD(learning_rate=LR_START)
        _, hist = train_model(ds_train, ds_test, 'cnn', opt, [callbacks])
        lrs = LR_START * (10 ** (np.arange(N_EPOCHS) / 20))
        plt.semilogx(lrs, hist.history["loss"])
        plt.axis([LR_START, LR_START * (10 ** (N_EPOCHS / 20)), 0, 20])
        plt.show()

    if FIND_BEST_MODEL:
        # (2b) train all models
        results = defaultdict(dict)
        for km, vm in MODELS.items():
            for ko, vo in OPTIMIZERS.items():
                key = km + '_' + ko + '_' + str(int(time()))
                results[key]['setting'] = 'model=' + km + ', opt=' + ko
                print('\n ********** run_' + results[key]['setting'] + ' **********')
                start = time()
                # vw=20
                results[key]['model'], results[key]['history'] = train_model(ds_train, ds_test, vm, vo, WINDOW_SIZE)
                # results[key]['model'], results[key]['history'] = train_model(ds_train, ds_test, vm, vo, vw)
                results[key]['training_time'] = np.round(time() - start, 3)
                y_pred = results[key]['model'].predict(ds_test).flatten()
                y_test = temps[split + WINDOW_SIZE:]
                mae = np.round(tf.keras.metrics.mae(y_test, y_pred).numpy(), 5)
                results[key]['mae'] = mae

        # (3) plot and print results
        print('\n ********** results: ********** ')
        ps = 20
        for i, k in enumerate(results.keys()):
            print(results[k]['setting'] + ', training_time=' + str(results[k]['training_time']) + ', mae=' + str(results[k]['mae']))
            plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['loss'][ps:], color=COLORS[i], alpha=.7)
            plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['val_loss'][ps:], linestyle='dashed',
                     color=COLORS[i], label=k, alpha=.7)
        plt.legend()
        plt.title('training loss min temp timeseries')
        plt.savefig('results/loss_hp.png')
        print('\n')

        # (4) save best model:
        #todo: seems to save the wrong model
        mae_ = {k: results[k]['mae'] for k in results.keys()}
        model_name = [k for k, v in mae_.items() if v == min(mae_.values())][0]
        results[model_name]['model'].save('saved_models/' + model_name)
        print('\n')
        print('********** saved: {} ********** '.format(model_name))


if __name__ == '__main__':
    main()
