import tensorflow as tf

L2 = tf.keras.regularizers.L2(0.01)


class Model:
    def __init__(self, model_type, input_shape):
        self.input_shape = input_shape
        self.model = None
        if model_type == 'cnn':
            self.model = self.model_cnn()

    def get_model(self):
        return self.model

    def model_cnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(40, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(40, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])


