
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict

from tfutils import model_checkpoint_callback, clear_path
from models import m30_imdb


NUM_WORDS = 5000
EMBED_SIZE = 20
MAX_LEN = 80
N_EPOCHS = 10

FIND_LR = False
LR_START = 1.0e-6

FIND_BEST_MODEL = True
COLORS = ['grey', 'blue', 'green', 'black', 'orange', 'red']

MODELS = {
    'simple': 'simple',
    # 'cnn': 'cnn',
    # 'lstm1': 'lstm1',
    # 'lstm2': 'lstm2',
    # 'gru1': 'gru1',
    # 'gru_do': 'gru_do',
    'lstm_bd': 'lstm_bd'
}
OPTIMIZERS = {
    'adam04': tf.keras.optimizers.Adam(lr=1e-4),
    # 'adam05': tf.keras.optimizers.Adam(lr=5e-4),
    # 'sgd05': tf.keras.optimizers.SGD(lr=5e-5)
}


def ds_to_texts_and_labels(ds):
    # necessary for tokenization
    df = tfds.as_dataframe(ds)
    sentences = df['text'].to_list()
    return [s.decode('utf-8') for s in sentences], np.array(df['label'])


def get_data(num_words, max_len):
    (train_ds, test_ds), info = tfds.load('imdb_reviews', split=['train', 'test'], with_info=True)

    train_sentences, train_labels = ds_to_texts_and_labels(train_ds)
    test_sentences, test_labels = ds_to_texts_and_labels(test_ds)
    tokenizer = Tokenizer(num_words=num_words, lower=True, oov_token='oov')
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    return train_sequences, train_labels, (test_sequences, test_labels)


def train_model(train_seq, train_labels, test_data, model, opt, callbacks=[]):
    model = m30_imdb.Model(model, NUM_WORDS, EMBED_SIZE).get_model()
    checkpoint_filepath = clear_path('/tmp/checkpoint_weights')
    callbacks_ = callbacks.copy()
    callbacks_.append(model_checkpoint_callback(checkpoint_filepath=checkpoint_filepath, monitor='val_loss', mode='min'))
    model.summary()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(train_seq, train_labels, validation_data=test_data, epochs=N_EPOCHS, callbacks=callbacks_,
                        verbose=1)
    model.load_weights(checkpoint_filepath)
    return model, history


def main():
    tf.keras.backend.clear_session()

    # (1) prepare data
    train_sequences, train_labels, test_data = get_data(NUM_WORDS, MAX_LEN)

    # (2a) find learning rate
    if FIND_LR:
        callbacks = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LR_START * 10**(epoch/20))
        opt = tf.keras.optimizers.Adam(learning_rate=LR_START)
        # opt = tf.keras.optimizers.SGD(learning_rate=LR_START)
        _, hist = train_model(train_sequences, train_labels, test_data, 'simple', opt, [callbacks])
        lrs = LR_START * (10 ** (np.arange(N_EPOCHS) / 10))
        plt.semilogx(lrs, hist.history["loss"])
        plt.axis([LR_START, LR_START * (10 ** (N_EPOCHS / 10)), 0, 20])
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
                results[key]['model'], results[key]['history'] = train_model(train_sequences, train_labels, test_data,
                                                                             vm, vo)
                results[key]['training_time'] = np.round(time() - start, 3)
                _, results[key]['acc'] = results[key]['model'].evaluate(test_data[0], test_data[1])

        # (3) plot and print results
        print('\n ********** results: ********** ')
        ps = 0
        for i, k in enumerate(results.keys()):
            print(results[k]['setting'] + ', training_time=' + str(results[k]['training_time']) + ', acc=' + str(results[k]['acc']))
            plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['loss'][ps:], color=COLORS[i], alpha=.7)
            plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['val_loss'][ps:], linestyle='dashed',
                     color=COLORS[i], label=k, alpha=.7)
        plt.legend()
        plt.title('training loss imdb')
        plt.savefig('results/loss_hp.png')
        print('\n')

        # (4) save best model:
        acc_ = {k: results[k]['acc'] for k in results.keys()}
        model_name = [k for k, v in acc_.items() if v == max(acc_.values())][0]
        results[model_name]['model'].save('saved_models/' + model_name)
        print('\n')
        print('********** saved: {} ********** '.format(model_name))


if __name__ == '__main__':
    main()
