
import numpy as np
import pandas as pd


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

    return data


def split_data(data, val_split, seed):
    '''split the data into training/test sets'''
    # extract demographic and preference values
    data_vals = data.values.copy()
    np.random.seed(seed)
    np.random.shuffle(data_vals)
    x,y = np.split(data_vals, [-5], axis=1)
    y /= 90

    # split into training and test sets
    ind_split = int(val_split * len(x))
    x_tr, x_ts = x[:-ind_split], x[-ind_split:]
    y_tr, y_ts = y[:-ind_split], y[-ind_split:]

    return (x_tr,y_tr), (x_ts,y_ts)

