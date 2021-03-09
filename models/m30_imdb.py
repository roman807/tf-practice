import tensorflow as tf


class Model:
    def __init__(self, model_type, vocab_size, embed_size=100):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.model = None
        if model_type == 'rnn':
            self.model = self.model_rnn()

    def get_model(self):
        return self.model

    def model_rnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])