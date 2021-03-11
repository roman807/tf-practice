import tensorflow as tf


class Model:
    def __init__(self, model_type, vocab_size, embed_size=100):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.model = None
        if model_type == 'simple':
            self.model = self.model_simple()
        if model_type == 'cnn':
            self.model = self.model_cnn()
        if model_type == 'lstm1':
            self.model = self.model_lstm1()
        if model_type == 'lstm2':
            self.model = self.model_lstm2()
        if model_type == 'gru1':
            self.model = self.model_gru1()
        if model_type == 'lstm_bd':
            self.model = self.model_lstm_bd()

    def get_model(self):
        return self.model

    def model_simple(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def model_cnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.Conv1D(60, 5, activation='relu'),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def model_gru1(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.GRU(50, activation='tanh', return_sequences=False),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def model_lstm1(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def model_lstm2(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(25, activation='tanh', return_sequences=False),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def model_lstm_bd(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(60, activation='tanh', return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, activation='tanh', return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])