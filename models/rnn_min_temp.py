import tensorflow as tf


def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), input_shape=[5]),
        # tf.keras.layers.Reshape((-1, 1), input_shape=[None]),
        tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
        # tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1),
        # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
        #                        input_shape=[None]),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 10.0)
    ])
