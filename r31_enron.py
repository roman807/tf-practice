# enron email dataset from tensorflow_datasets, predict next word
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from models import m31_enron

NUM_WORDS = 1000
EMBED_SIZE = 50
MAX_LEN = 10
N_EPOCHS = 1

N_EMAILS_TRAIN = 4_000
N_EMAILS_TEST = 1_000
MAX_LEN_EMAIL = 40


def ds_to_texts(ds, n_emails):
    sentences = []
    for i, email in enumerate(ds):
        if i >= n_emails:
            continue
        sentences.append(email['email_body'].numpy().decode('utf-8'))
    return sentences


def main():
    # (1) prepare data
    (train_ds, test_ds), info = tfds.load('aeslc', split=['train', 'test'], with_info=True)
    train_sentences = ds_to_texts(train_ds, N_EMAILS_TRAIN)
    test_sentences = ds_to_texts(test_ds, N_EMAILS_TEST)
    tokenizer = Tokenizer(NUM_WORDS, lower=True, oov_token='oov')
    tokenizer.fit_on_texts(train_sentences)
    train_seq = tokenizer.texts_to_sequences(train_sentences)
    test_seq = tokenizer.texts_to_sequences(test_sentences)

    input_sequences = []
    for seq in train_seq:
        for i in range(1, MAX_LEN_EMAIL - 1):
            input_sequences.append(seq[:i+1])

    train_seq = np.array(pad_sequences(input_sequences, maxlen=MAX_LEN, padding='pre', truncating='pre'))
    x_train, y_train = train_seq[:, :-1], train_seq[:, -1]

    model = m31_enron.Model('lstm1', NUM_WORDS, EMBED_SIZE).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=20, verbose=1)

    #todo: check softmax (should predict on 1000 classes)
    text = ['i', 'hope', 'you']
    for i in range(30):
        input_ = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding='pre', truncating='pre')[0]
        text.append(tokenizer.sequences_to_texts(model.predict_classes(input_)))
    print(text)


if __name__ == '__main__':
    main()

