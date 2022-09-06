
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

from besci import besci_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fix_random_seed(seed):
    '''fix random seed for reproducibility'''
    tf.random.set_seed(seed)
    np.random.seed(seed)


def setup_model(input_size, output_size, arch=[], seed=0):
    '''create deep learning model'''
    fix_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_size),
        *[tf.keras.layers.Dense(l, activation='relu') for l in arch],
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model


def normalize(data):
    '''normalize the provided data to be on the same scale'''
    return np.array(data) / data[0]


def train_model_ml(model, params, train_data, test_data):
    '''train model with conventional machine learning'''
    x_tr, y_tr = train_data
    x_ts, y_ts = test_data
    fix_random_seed(params['seed'])
    optimizer = tf.keras.optimizers.SGD()

    # compile and train ml-model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalCrossentropy()])
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    history = model.fit(x_tr, y_tr, batch_size=params['batch_size'], epochs=params['num_epoch'],
                        validation_data=(x_ts,y_ts), verbose=0, callbacks=[tqdm_callback])

    # extract loss and metrics
    loss_tr = normalize(history.history['loss'])
    loss_ts = normalize(history.history['val_loss'])
    mse_tr = normalize(history.history['mean_squared_error'])
    mse_ts = normalize(history.history['val_mean_squared_error'])
    ent_tr = normalize(history.history['categorical_crossentropy'])
    ent_ts = normalize(history.history['val_categorical_crossentropy'])

    return (loss_tr, loss_ts), (mse_tr, mse_ts), (ent_tr, ent_ts)


def train_model_bs(model, params, train_data, test_data):
    '''train model with behavioral science and machine learning'''
    x_tr, y_tr = train_data
    x_ts, y_ts = test_data
    fix_random_seed(params['seed'])
    optimizer = tf.keras.optimizers.SGD()

    # compile and train bs-model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalCrossentropy()])
    loss_tr, loss_ts, mse_tr, mse_ts, ent_tr, ent_ts = [], [], [], [], [], []
    for epoch in tqdm(range(params['num_epoch']), desc='Training'):
        batch_losses = []

        # split data into batches
        inds = np.arange(len(x_tr))
        np.random.shuffle(inds)
        num_batch = np.ceil(len(x_tr) / params['batch_size']).astype(int)
        for ind in np.array_split(inds, num_batch):

            # compute loss over the batch
            x, y = x_tr[ind], y_tr[ind]
            with tf.GradientTape() as tape:
                z = model(x)
                loss = besci_loss(x, y, z)
                batch_losses.append(loss.numpy())
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # compute loss and metrics over the epoch
        loss_tr.append(np.mean(batch_losses))
        loss_ts.append(besci_loss(x_ts, y_ts, model(x_ts)).numpy())
        mse_tr.append(tf.keras.metrics.MeanSquaredError()(y_tr, model(x_tr)))
        mse_ts.append(tf.keras.metrics.MeanSquaredError()(y_ts, model(x_ts)))
        ent_tr.append(tf.keras.metrics.CategoricalCrossentropy()(y_tr, model(x_tr)))
        ent_ts.append(tf.keras.metrics.CategoricalCrossentropy()(y_ts, model(x_ts)))

    # normalize loss and metrics
    loss_tr = normalize(loss_tr)
    loss_ts = normalize(loss_ts)
    mse_tr = normalize(mse_tr)
    mse_ts = normalize(mse_ts)
    ent_tr = normalize(ent_tr)
    ent_ts = normalize(ent_ts)

    return (loss_tr, loss_ts), (mse_tr, mse_ts), (ent_tr, ent_ts)

