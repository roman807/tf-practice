# for reference: https://www.coursera.org/learn/natural-language-processing-tensorflow/lecture/0N8WC/looking-into-the-details
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from models import m30_imdb

NUM_WORDS = 1000
EMBED_SIZE = 50
MAX_LEN = 20
N_EPOCHS = 2


def ds_to_texts_and_labels(ds):
    # necessary for tokenization
    df = tfds.as_dataframe(ds)
    sentences = df['text'].to_list()
    return [s.decode('utf-8') for s in sentences], np.array(df['label'])


def main():
    # (1) prepare data
    (train_ds, test_ds), info = tfds.load('imdb_reviews', split=['train', 'test'], with_info=True)

    train_sentences, train_labels = ds_to_texts_and_labels(train_ds)
    test_sentences, test_labels = ds_to_texts_and_labels(test_ds)
    tokenizer = Tokenizer(num_words=NUM_WORDS, lower=True, oov_token='oov')
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_sequences = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_sequences = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    model = m30_imdb.Model('rnn', NUM_WORDS, EMBED_SIZE).get_model()
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=N_EPOCHS,
              batch_size=20, verbose=1)
    _, acc = model.evaluate(test_sequences, test_labels)


if __name__ == '__main__':
    main()
