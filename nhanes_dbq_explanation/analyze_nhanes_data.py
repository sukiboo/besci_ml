
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from process_nhanes_data import load_data, encode_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class NhanesDataAnalyzer:
    '''learn and explain the data from NHANES surveys'''

    def __init__(self, nhanes_data, random_seed):
        self.fix_random_seed(random_seed)
        self.get_data(nhanes_data)
        self.split_data(val_split=0.1)
        self.get_model()

    def fix_random_seed(self, random_seed):
        '''fix random seed for reproducibility'''
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    def get_data(self, nhanes_data):
        '''load and encode NHANES data'''
        self.data = pd.concat(map(lambda x: load_data(*x), nhanes_data.values()), ignore_index=True)
        self.data, self.features, self.outputs = encode_data(self.data)

    def split_data(self, val_split):
        '''split the data into training/test sets'''
        # extract demographic and preference values
        data_vals = self.data.values.copy()
        np.random.shuffle(data_vals)
        x = data_vals[:,:len(self.features)]
        y = data_vals[:,len(self.features):]
        # split into training and test sets
        ind_split = int(val_split * len(x))
        self.x_tr, self.x_ts = x[:-ind_split], x[-ind_split:]
        self.y_tr, self.y_ts = y[:-ind_split], y[-ind_split:]

    def get_model(self):
        '''create and train deep learning model'''
        # create model
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=len(self.features)),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(len(self.outputs), activation=None)])

        # compile and train model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
        tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
        history = self.model.fit(self.x_tr, self.y_tr, batch_size=32, epochs=100, verbose=0,
                                 validation_data=(self.x_ts, self.y_ts), callbacks=[tqdm_callback])

        # report training and test losses
        loss = history.history['loss'], history.history['val_loss']
        print(f'model is trained: loss = {loss[0][-1]:.2e} / {loss[1][-1]:.2e}')
        self.plot_loss(loss, show=True)

    def plot_loss(self, loss, show=True):
        '''plot training and test losses'''
        sns.set_palette('tab20')
        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(loss[0], linewidth=3, label='train loss')
        plt.plot(loss[1], linestyle='--', linewidth=3, label='test loss')
        ##plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'./images/model_loss.png', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def analyze(self):
        '''analyze the model'''
        for i in range(len(self.outputs)):
            pred = lambda x: self.model.predict(x)[:,i]
            self.plot_shapley(pred, name=f'shapley_{self.outputs[i]}')

    def plot_shapley(self, pred, name=None):
        '''compute and plot shapley values for a given prediction function'''
        explainer = shap.Explainer(pred, self.x_tr,
                                   feature_names=self.features,
                                   output_names=self.outputs)
        shap_values = explainer(self.x_ts)
        shap.plots.beeswarm(shap_values, order=np.arange(len(self.features)), plot_size=(8,4), show=False)
        plt.xlim((-.1,.1))
        plt.tight_layout()
        if name is not None:
            plt.savefig(f'./images/{name}.png', dpi=300)
            plt.close()
        else:
            plt.show()

