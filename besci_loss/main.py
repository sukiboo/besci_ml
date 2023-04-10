'''
    this is a prototype code for the 'Behavioral ML' project
    the main thing that needs to be changed is the besci.py file that defines the 'behavioral loss'

    note to self: make sure to reorganize this code before the submission, this is embarassing

    UPD: this direction is stuck because my hypothesis that utilizing behavioral science
    by modifying the loss function seems to not be correct (at least in its current formulation)
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from besci import besci_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style='darkgrid', palette='Paired', font='monospace')


def fix_random_seed(seed=2022):
    '''fix random seed for reproducibility'''
    tf.random.set_seed(seed)
    np.random.seed(seed)


def load_data(data_file, validation_split=.1):
    '''preprocess data and split it into training/test sets'''
    # load and preprocess
    data = pd.read_csv(data_file)
    data['age'] = (data['age'] - 20) / 60
    data['income'] /= 5
    data['num_meals_not_home'] *= 30/7
    data['num_meals_fast_food'] *= 30/7
    data.drop(data[data['num_frozen_foods_30_days']\
                 + data['num_ready_foods_30_days']\
                 + data['num_meals_not_home']\
                 + data['num_meals_fast_food'] > 90].index, inplace=True)
    data['num_home_cook'] = 90 - data['num_meals_not_home']\
                               - data['num_meals_fast_food']\
                               - data['num_ready_foods_30_days']\
                               - data['num_frozen_foods_30_days']

    # extract demographic and preference values
    fix_random_seed()
    data_vals = data.values.copy()
    np.random.shuffle(data_vals)
    x,y = np.split(data_vals, [-5], axis=1)
    y /= 90

    # split into training and test sets
    ind_split = int(validation_split * len(x))
    x_tr, x_ts = x[:-ind_split], x[-ind_split:]
    y_tr, y_ts = y[:-ind_split], y[-ind_split:]

    return (x_tr,y_tr), (x_ts,y_ts), data


def setup_model(input_size, output_size):
    '''create deep learning model'''
    fix_random_seed()
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_size),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model


def plot_losses(loss_tr, loss_ts, normalize=True, name=None):
    '''plot training and test losses'''
    fig, ax = plt.subplots(figsize=(8,5))
    if normalize:
        loss_tr = np.array(loss_tr) / loss_tr[0]
        loss_ts = np.array(loss_ts) / loss_ts[0]
    plt.plot(loss_tr, linewidth=3)
    plt.plot(loss_ts, linewidth=3)
    plt.legend(['training loss', 'test loss'], loc='upper right')
    plt.title('Training and test losses')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/losses_{name}.png', dpi=300, format='png')
    plt.show()


def plot_losses2(losses_tr, losses_ts, normalize=True, name=None):
    '''plot training and test losses'''
    fig, ax = plt.subplots(figsize=(8,5))
    if normalize:
        losses_tr[0] = np.array(losses_tr[0]) / losses_tr[0][0]
        losses_tr[1] = np.array(losses_tr[1]) / losses_tr[1][0]
        losses_ts[0] = np.array(losses_ts[0]) / losses_ts[0][0]
        losses_ts[1] = np.array(losses_ts[1]) / losses_ts[1][0]
    plt.plot(losses_tr[0], linewidth=3)
    plt.plot(losses_ts[0], linestyle='--', linewidth=3)
    plt.plot(losses_tr[1], linewidth=3)
    plt.plot(losses_ts[1], linestyle='--', linewidth=3)
    plt.legend(['training loss', 'test loss'], loc='upper right')
    plt.title('Training and test losses')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/losses_{name}.png', dpi=300, format='png')
    plt.show()


def plot_metrics(metric_tr, metric_ts, normalize=True, name=None):
    '''plot training and test metrics'''
    fig, ax = plt.subplots(figsize=(8,5))
    if normalize:
        metric_tr = np.array(metric_tr) / metric_tr[0]
        metric_ts = np.array(metric_ts) / metric_ts[0]
    plt.plot(metric_tr, linewidth=3)
    plt.plot(metric_ts, linewidth=3)
    plt.legend(['training metric', 'test metric'], loc='upper right')
    plt.title('Training and test metrics')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/metrics_{name}.png', dpi=300, format='png')
    plt.show()


def plot_metrics2(metrics_tr, metrics_ts, normalize=True, name=None):
    '''plot training and test metrics'''
    fig, ax = plt.subplots(figsize=(8,5))
    if normalize:
        metrics_tr[0] = np.array(metrics_tr[0]) / metrics_tr[0][0]
        metrics_tr[1] = np.array(metrics_tr[1]) / metrics_tr[1][0]
        metrics_ts[0] = np.array(metrics_ts[0]) / metrics_ts[0][0]
        metrics_ts[1] = np.array(metrics_ts[1]) / metrics_ts[1][0]
    plt.plot(metrics_tr[0], linewidth=3)
    plt.plot(metrics_ts[0], linestyle='--', linewidth=3)
    plt.plot(metrics_tr[1], linewidth=3)
    plt.plot(metrics_ts[1], linestyle='--', linewidth=3)
    plt.legend(['training metric', 'test metric'], loc='upper right')
    plt.title('Training and test metrics')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/metrics_{name}.png', dpi=300, format='png')
    plt.show()


def train_model_ml(model, params):
    '''train model with conventional machine learning'''
    fix_random_seed()
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    # compile and train ml-model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.CategoricalCrossentropy())
    # model.compile(optimizer=optimizer,
                  # loss=tf.keras.losses.CategoricalCrossentropy(),
                  # metrics=tf.keras.metrics.MeanSquaredError())
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    history = model.fit(x_tr, y_tr, batch_size=params['batch_size'], epochs=params['num_epoch'],
                        validation_data=(x_ts,y_ts), verbose=0, callbacks=[tqdm_callback])

    return (history.history['loss'], history.history['val_loss']),\
           (history.history['categorical_crossentropy'], history.history['val_categorical_crossentropy'])
           # (history.history['mean_squared_error'], history.history['val_mean_squared_error'])


def train_model_bs(model, params):
    '''train model with behavioral science and machine learning'''
    fix_random_seed()
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    # compile and train bs-model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.CategoricalCrossentropy())
    # model.compile(optimizer=optimizer,
                  # loss=tf.keras.losses.CategoricalCrossentropy(),
                  # metrics=tf.keras.metrics.MeanSquaredError())
    losses_tr, losses_ts = [], []
    metrics_tr, metrics_ts = [], []
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

        # compute loss over the epoch
        losses_tr.append(np.mean(batch_losses))
        losses_ts.append(besci_loss(x_ts, y_ts, model(x_ts)).numpy())
        metrics_tr.append(tf.keras.metrics.CategoricalCrossentropy()(y_tr, model(x_tr)))
        metrics_ts.append(tf.keras.metrics.CategoricalCrossentropy()(y_ts, model(x_ts)))
        # metrics_tr.append(tf.keras.metrics.MeanSquaredError()(y_tr, model(x_tr)))
        # metrics_ts.append(tf.keras.metrics.MeanSquaredError()(y_ts, model(x_ts)))

    return (losses_tr, losses_ts), (metrics_tr, metrics_ts)


def plot_dem_sensitivity(model_ml, model_bs, dem_ind):
    x = x_ts.copy()
    dists_ml, dists_bs = [], []
    for d in np.linspace(0,1,11):
        x[:,dem_ind] = d
        dists_ml.append(model_ml.predict(x).mean(axis=0))
        dists_bs.append(model_bs.predict(x).mean(axis=0))
    diffs_ml = [sum(abs(dists_ml[i+1] - dists_ml[i])) for i in range(len(dists_ml)-1)]
    diffs_bs = [sum(abs(dists_bs[i+1] - dists_bs[i])) for i in range(len(dists_bs)-1)]
    # plot demographic sensitivity
    plt.plot(diffs_ml)
    plt.plot(diffs_bs)
    plt.show()


if __name__ == '__main__':

    # get train/test data
    (x_tr,y_tr), (x_ts,y_ts), data = load_data('./data/wain_dataset_one_hot.csv', validation_split=.1)
    input_size, output_size = x_tr.shape[1], y_tr.shape[1]

    # create ml and bs models
    model_ml = setup_model(input_size, output_size)
    model_bs = setup_model(input_size, output_size)

    # train ml and bs models
    params = {'batch_size': 32, 'num_epoch': 100, 'learning_rate': 1e-4}
    losses_ml, metrics_ml = train_model_ml(model_ml, params)
    losses_bs, metrics_bs = train_model_bs(model_bs, params)

    # plot losses and metrics
    plot_losses(*losses_ml, name='ml')
    plot_metrics(*metrics_ml, name='ml')
    plot_losses(*losses_bs, name='bs')
    plot_metrics(*metrics_bs, name='bs')
    plot_losses2([losses_ml[0], losses_bs[0]], [losses_ml[1], losses_bs[1]], name='losses')
    plot_metrics2([metrics_ml[0], metrics_bs[0]], [metrics_ml[1], metrics_bs[1]], name='metrics')

    for dem_ind in [0,1]:
        plot_dem_sensitivity(model_ml, model_bs, dem_ind)

