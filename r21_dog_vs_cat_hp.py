import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from time import time
import matplotlib.pyplot as plt
from collections import defaultdict

from models import m21_cnn_dog_vs_cat
from tfutils import model_checkpoint_callback, clear_path


N_EPOCHS = 25
COLORS = ['grey', 'blue', 'green', 'black', 'orange', 'red']
MODELS = {'cnn2': 'cnn2'} #, 'cnn_large': 'cnn_large'}
AUGMENTATIONS = {'no_aug': False, 'aug': True}

TARGET_ACC_CALLBACK = .9
TARGET_ACC_SAVE = .8


def get_data(augment):
    train_dir = 'data/dog_vs_cat/dataset/training_set'
    test_dir = 'data/dog_vs_cat/dataset/test_set'
    data_gen = ImageDataGenerator(rescale=1 / 255.0)
    if augment:
        data_gen_train = ImageDataGenerator(rescale=1 / 255.0, rotation_range=40, zoom_range=.2, shear_range=.2,
                                            width_shift_range=.2, height_shift_range=.2, vertical_flip=True)
        train_data_gen = data_gen_train.flow_from_directory(train_dir, target_size=(28, 28), batch_size=20)
    else:
        train_data_gen = data_gen.flow_from_directory(train_dir, target_size=(28, 28), batch_size=20)
    test_data_gen = data_gen.flow_from_directory(test_dir, target_size=(28, 28), batch_size=40)
    return train_data_gen, test_data_gen


def train_model(train_data_gen, test_data_gen, model, opt, callbacks=[]):
    model = m21_cnn_dog_vs_cat.Model(model, input_shape=(28, 28, 3)).get_model()
    checkpoint_filepath = clear_path('/tmp/checkpoint_weights')
    callbacks_ = callbacks.copy()
    callbacks_.append(model_checkpoint_callback(checkpoint_filepath=checkpoint_filepath, monitor='val_acc', mode='max'))
    model.summary()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=N_EPOCHS, callbacks=callbacks_,
                        verbose=1)
    model.load_weights(checkpoint_filepath)
    return model, history


def main():
    # add regularization, use transfer learning
    # # confirm that GPU is used:
    # print(tf.config.list_physical_devices('GPU'))
    results = defaultdict(dict)
    for km, vm in MODELS.items():
        for ka, va in AUGMENTATIONS.items():
            key = km + '_' + ka + '_' + str(int(time()))
            results[key]['setting'] = 'model=' + km + ', aug=' + ka
            train_data_gen, test_data_gen = get_data(va)
            opt = tf.keras.optimizers.Adam(lr=1.0e-3)
            start = time()
            results[key]['model'], results[key]['history'] = train_model(train_data_gen, test_data_gen, vm, opt)
            results[key]['training_time'] = np.round(time() - start, 3)
            _, results[key]['acc'] = results[key]['model'].evaluate(test_data_gen)

    # (3) plot and print results
    print('\n ********** results: ********** ')
    ps = 0
    for i, k in enumerate(results.keys()):
        print(results[k]['setting'] + ', training_time=' + str(results[k]['training_time']) + ', acc=' + str(results[k]['acc']))
        plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['loss'][ps:], color=COLORS[i], alpha=.7)
        plt.plot(results[k]['history'].epoch[ps:], results[k]['history'].history['val_loss'][ps:], linestyle='dashed',
                 color=COLORS[i], label=k, alpha=.7)
    plt.legend()
    plt.title('training loss dogs vs cats')
    plt.savefig('results/loss_dvc_hp1.png')
    print('\n')

    # (4) save best model:
    acc_ = {k: results[k]['acc'] for k in results.keys()}
    model_name = [k for k, v in acc_.items() if v == max(acc_.values())][0]
    results[model_name]['model'].save('saved_models/' + model_name)
    print('\n')
    print('********** saved: {} ********** '.format(model_name))


if __name__ == '__main__':
    main()