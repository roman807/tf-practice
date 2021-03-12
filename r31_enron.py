# enron email dataset from tensorflow_datasets, predict next word
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

from models import m31_enron

NUM_WORDS = 5000
EMBED_SIZE = 50
MAX_LEN = 10
N_EPOCHS = 3

MODEL = 'lstm1'
N_EMAILS_TRAIN = 4_000
N_EMAILS_TEST = 1_000
MAX_LEN_EMAIL = 30


def ds_to_texts(ds, n_emails):
    sentences = []
    for i, email in enumerate(ds):
        if i >= n_emails:
            continue
        sentences.append(email['email_body'].numpy().decode('utf-8'))
    return sentences


def get_input_sequences(sentences, tokenizer):
    seq = tokenizer.texts_to_sequences(sentences)
    input_sequences = []
    for seq in seq:
        for i in range(1, min(len(seq)+1, MAX_LEN_EMAIL) - 1):
            input_sequences.append(seq[:i+1])
    seq = np.array(pad_sequences(input_sequences, maxlen=MAX_LEN, padding='pre', truncating='pre'))
    return seq[:, :-1], seq[:, -1]


def generate_text(text, tokenizer, model):
    for i in range(30):
        input_ = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding='pre', truncating='pre')
        prediction = model.predict(input_)[0]
        token = np.random.choice(len(prediction), p=prediction)
        text.append(tokenizer.sequences_to_texts([[token]])[0])
        # text.append(tokenizer.sequences_to_texts([np.argmax(model.predict(input_), axis=-1)])[0])
    print(' '.join(text))


def main():
    # (1) prepare data
    (train_ds, test_ds), info = tfds.load('aeslc', split=['train', 'test'], with_info=True)
    train_sentences = ds_to_texts(train_ds, N_EMAILS_TRAIN)
    test_sentences = ds_to_texts(test_ds, N_EMAILS_TEST)
    # tokenizer = Tokenizer(NUM_WORDS, lower=True, oov_token='oov')
    tokenizer = Tokenizer(NUM_WORDS, lower=True)
    tokenizer.fit_on_texts(train_sentences)

    x_train, y_train = get_input_sequences(train_sentences, tokenizer)
    x_test, y_test = get_input_sequences(test_sentences, tokenizer)

    model = m31_enron.Model(MODEL, NUM_WORDS, EMBED_SIZE).get_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=1e-2)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=N_EPOCHS, batch_size=20, verbose=1)

    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'], color='orange')
    plt.show()

    generate_text(['i', 'hope', 'you'], tokenizer, model)
    generate_text(['try', 'something'], tokenizer, model)
    generate_text(['please'], tokenizer, model)
    generate_text(['please', 'follow'], tokenizer, model)
    generate_text(['please', 'follow', 'up', 'with'], tokenizer, model)
    generate_text(['why'], tokenizer, model)


if __name__ == '__main__':
    main()
