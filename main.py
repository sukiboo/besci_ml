
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style='darkgrid', palette='tab10', font='monospace')


def load_data(data_file):
    '''load and preprocess the data'''
    data = pd.read_csv(data_file)

    # preprocess the data
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

    data[data.columns[-5:]] /= 90

    return data


def split_data(data, val_split, seed):
    '''split the data into training/test sets'''
    # extract demographic and preference values
    data_vals = data.values.copy()
    np.random.seed(seed)
    np.random.shuffle(data_vals)
    x,y = np.split(data_vals, [11], axis=1)

    # split into training and test sets
    ind_split = int(val_split * len(x))
    x_tr, x_ts = x[:-ind_split], x[-ind_split:]
    y_tr, y_ts = y[:-ind_split], y[-ind_split:]

    return (x_tr,y_tr), (x_ts,y_ts)


def fix_random_seed(seed):
    '''fix random seed for reproducibility'''
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_model(train_data, test_data, seed):
    '''create and train deep learning model'''
    fix_random_seed(seed)

    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=train_data[0].shape[1]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(train_data[1].shape[1], activation='softmax')
    ])

    # compile and train model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    loss_tr = [model.evaluate(*train_data, verbose=0)]
    loss_ts = [model.evaluate(*test_data, verbose=0)]
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    history = model.fit(*train_data, batch_size=32, epochs=100,
                        validation_data=test_data, verbose=0, callbacks=[tqdm_callback])

    # extract loss and metrics
    loss_tr += history.history['loss']
    loss_ts += history.history['val_loss']

    return model, (loss_tr,loss_ts)


def plot_loss(loss):
    '''plot training and test losses'''
    sns.set_palette('tab20')
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(loss[0], linewidth=3, label='train loss')
    plt.plot(loss[1], linestyle='--', linewidth=3, label='test loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'./loss.png', dpi=300, format='png')
    plt.show()


def synthesize_demographics(num, seed):
    '''generate synthetic people'''
    # generate random values
    np.random.seed(seed)
    dems = {'age': np.random.rand(num),
             'income': np.random.rand(num),
             'gender_male': np.random.randint(2, size=num),
             'education': np.random.randint(5, size=num) / 4.,
             'race': np.random.randint(6, size=num),
             'marital_status': np.random.randint(3, size=num)
            }

    # encode synthetic people
    synth_dems = pd.DataFrame(dems)
    synth_dems = pd.get_dummies(synth_dems, columns=['race'], drop_first=True)
    synth_dems = pd.get_dummies(synth_dems, columns=['marital_status'], drop_first=True)
    synth_dems.columns = data.columns[:-5]

    return synth_dems


def synthesize_preferences(dems, seed):
    '''generate synthetic food preferences'''
    np.random.seed(seed)
    pref = model(dems.values).numpy()
    noise = 0.1 * np.random.randn(*pref.shape)
    pert = (.8 - .5*dems['age'] + .2*dems['gender_male'] - .2*dems['education']).values.reshape(-1,1)
    synth_pref = np.abs(pref + pert * noise)
    synth_pref /= synth_pref.sum(axis=1, keepdims=True)
    synth_pref = pd.DataFrame(synth_pref, columns=['num_meals_not_home', 'num_meals_fast_food',
                              'num_ready_foods_30_days', 'num_frozen_foods_30_days', 'num_home_cook'])
    return synth_pref


if __name__ == '__main__':

    # get train/test data
    data = load_data('./data/wain_dataset_one_hot.csv')
    train_data, test_data = split_data(data, val_split=0.1, seed=2022)

    # train ml model
    model, loss = get_model(train_data, test_data, seed=2022)
    print(f'model is trained: loss = {loss[0][-1]:.2e} / {loss[1][-1]:.2e}')
    plot_loss(loss)

    # extrapolate data
    dems = synthesize_demographics(num=10000, seed=2022)
    pref = synthesize_preferences(dems, seed=2022)
    print(f'data is synthesized, average deviation = {np.abs(pref.values - model(dems.values).numpy()).mean():.4f}')
    data_synth = pd.concat([dems, pref], axis=1)

    # save data
    data.to_csv('./data/original.csv', index=False)
    data_synth.to_csv('./data/synthetic.csv', index=False)



