import tensorflow as tf


def windowed_dataset(series, window_size=5, batch_size=20, shuffle=True, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size +1))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric='accuracy', greater=True, target=.99):
        super().__init__()
        self.metric = metric
        self.greater = greater
        self.target = target

    def on_epoch_end(self, epoch, logs={}):
        if self.greater:
            if logs.get(self.metric) > self.target:
                print('\n stop training reached {} {}'.format(self.target, self.metric))
                self.model.stop_training = True
        else:
            if logs.get(self.metric) < self.target:
                print('\n stop training reached {} {}'.format(self.target, self.metric))
                self.model.stop_training = True
