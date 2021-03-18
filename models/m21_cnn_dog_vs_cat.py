import tensorflow as tf

L2 = tf.keras.regularizers.L2(0.01)


class Model:
    def __init__(self, model_type, input_shape):
        self.input_shape = input_shape
        self.model = None
        if model_type == 'cnn':
            self.model = self.model_cnn()
        if model_type == 'cnn2':
            self.model = self.model_cnn2()
        if model_type == 'cnn2_reg':
            self.model = self.model_cnn2_reg()
        if model_type == 'cnn_large':
            self.model = self.model_cnn_large()

    def get_model(self):
        return self.model

    def model_cnn(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(40, kernel_size=(5, 5), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(40, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    def model_cnn2(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(30, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(60, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    def model_cnn2_reg(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(30, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                                   kernel_regularizer=None),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(60, kernel_size=(3, 3), activation='relu', kernel_regularizer=None),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(80, activation='relu', kernel_regularizer=L2),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=L2)
        ])

    def model_cnn_large(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(80, kernel_size=(5, 5), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(80, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
