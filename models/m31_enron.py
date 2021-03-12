import tensorflow as tf


class Model:
    def __init__(self, model_type, vocab_size, embed_size=100):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.model = None
        if model_type == 'lstm1':
            self.model = self.model_lstm1()
        if model_type == 'lstm_bd2':
            self.model = self.model_lstm_bd2()

    def get_model(self):
        return self.model

    def model_lstm1(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])

    def model_lstm_bd2(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False)),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])