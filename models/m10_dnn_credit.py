import tensorflow as tf


class Model:
    def __init__(self, model_type, input_shape):
        self.input_shape = input_shape
        self.model = None
        if model_type == 'dnn':
            self.model = self.model_dnn()

    def get_model(self):
        return self.model

    def model_dnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(100, input_shape=self.input_shape, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
