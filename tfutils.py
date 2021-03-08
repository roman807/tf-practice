import tensorflow as tf
import os
import shutil


def windowed_dataset(series, window_size=5, batch_size=20, shuffle=True, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


class MyCallbackStopTraining(tf.keras.callbacks.Callback):
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


def model_checkpoint_callback(checkpoint_filepath, monitor, mode):
    # if not os.path.isdir(checkpoint_filepath):
    #     os.mkdir(checkpoint_filepath)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode=mode,
        save_best_only=True
    )


def clear_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    return path
