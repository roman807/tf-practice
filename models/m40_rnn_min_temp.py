import tensorflow as tf


class Model:
    def __init__(self, model_type, window_size):
        self.window_size = window_size
        self.model = None
        if model_type == 'rnn':
            self.model = self.model_rnn()
        elif model_type == 'cnn':
            self.model = self.model_cnn()
        elif model_type == 'cnn_rnn':
            self.model = self.model_cnn_rnn()

    def get_model(self):
        return self.model

    def model_rnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), input_shape=[self.window_size]),
            # tf.keras.layers.Reshape((-1, 1), input_shape=[None]),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 10.0)
        ])

    def model_cnn(self):
        # trains quicker, better results
        return tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), input_shape=[self.window_size]),
            tf.keras.layers.Conv1D(40, kernel_size=3, activation='relu'),
            tf.keras.layers.Conv1D(40, kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 10.0)
        ])

    def model_cnn_rnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), input_shape=[self.window_size]),
            tf.keras.layers.Conv1D(40, kernel_size=3, activation='relu'),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 10.0)
        ])
