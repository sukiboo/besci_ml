
import numpy as np
import pandas as pd


np.set_printoptions(precision=3, suppress=True)

def get_dem_data(demographics_file):
    '''load and encode demographic data'''
    df = pd.read_sas(f'./data/{demographics_file}')

    # columns to extract
    columns = {'id': 'SEQN',
               'age': 'RIDAGEYR',
               'gender': 'RIAGENDR',
               'income': 'INDFMPIR',
               'education': 'DMDEDUC2',
               'race': 'RIDRETH1',
               'marital': 'DMDMARTL',
               'household': 'DMDHHSIZ'}
    df = df[columns.values()]
    df.columns = columns.keys()

    # remove missing and incorrect values
    df = df.dropna()
    df = df[df.age.ge(20) & df.age.le(80) &
            df.gender.isin([1,2]) &
            df.income.ge(0.) & df.income.le(5.) &
            df.education.isin([1,2,3,4,5]) &
            df.race.isin([1,2,3,4,5,6,7]) &
            df.marital.isin([1,2,3,4,5,6]) &
            df.household.isin([1,2,3,4,5,6,7])]

    # encode demographic features
    df.age = (df.age - 20) / 60
    df.gender = 2 - df.gender
    df.income = df.income / 5
    df.education = (df.education - 1) / 4
    df.household = (df.household - 1) / 6
    race_dict = {1: .66, 2: .66, 3: 1., 4: .33, 5: 0.}
    marital_dict = {1: 1., 2: .5, 3: .5, 4: .5, 5: 0., 6: 1.}
    df = df.replace({'race': race_dict, 'marital': marital_dict})

    return df.reset_index(drop=True)


def get_dbq_data(dbq_file):
    '''load and encode dietary preferences data'''
    df = pd.read_sas(f'./data/{dbq_file}')

    # columns to extract
    columns = {'id': 'SEQN',
               'num_not_home_7': 'DBD895',
               'num_fast_food_7': 'DBD900',
               'num_ready_30': 'DBD905',
               'num_frozen_30': 'DBD910'}
    df = df[columns.values()]
    df.columns = columns.keys()

    # remove missing and incorrect values
    df = df.dropna()
    df = df[df.num_not_home_7.ge(0) & df.num_not_home_7.le(21) &
            df.num_fast_food_7.ge(0) & df.num_fast_food_7.le(21) &
            df.num_not_home_7.ge(df.num_fast_food_7) &
            df.num_ready_30.ge(0) & df.num_ready_30.le(90) &
            df.num_frozen_30.ge(0) & df.num_frozen_30.le(90) &
            (df.num_ready_30 + df.num_frozen_30).le(90)]

    # encode food preferences as probability
    df['eat_out'] = (df.num_not_home_7 - df.num_fast_food_7) / 21
    df['fast_food'] = df.num_fast_food_7 / 21
    df['ready'] = df.num_ready_30 / 90
    df['frozen'] = df.num_frozen_30 / 90
    df['cook'] = 1 - df.eat_out - df.fast_food - df.ready - df.frozen
    df = df.drop(columns=['num_not_home_7', 'num_fast_food_7', 'num_ready_30', 'num_frozen_30'])

    return df.reset_index(drop=True)


def get_alq_data(alq_file):
    '''load and encode alcohol consumption data'''
    df = pd.read_sas(f'./data/{alq_file}')

    # columns to extract
    columns = {'id': 'SEQN',
               'alcohol_freq': 'ALQ121',
               'alcohol_num': 'ALQ130'}
    df = df[columns.values()]
    df.columns = columns.keys()

    # remove missing and incorrect values
    df = df.dropna()
    df = df[df.alcohol_freq.isin([0,1,2,3,4,5,6,7,8,9,10]) &
            df.alcohol_num.ge(1) & df.alcohol_num.le(13)]

    # encode food preferences as probability
    df.alcohol_num = (df.alcohol_num - 1) / 12
    alcohol_freq_dict = {0: .0, 1: 1., 2: .9, 3: .8, 4: .7, 5: .6, 6: .5, 7: .4, 8: .3, 9: .2, 10: .1}
    df = df.replace({'alcohol_freq': alcohol_freq_dict})

    return df.reset_index(drop=True)


def make_train_data(inputs, outputs, val_split=.1, random_seed=0):
    '''split the data into training/test sets'''
    features = list(inputs.columns[1:])
    prefs = list(outputs.columns[1:])

    # merge demographic and preference values
    vals = inputs.merge(outputs, on='id').dropna().drop(columns=['id']).values
    np.random.seed(random_seed)
    np.random.shuffle(vals)
    x = vals[:,:len(features)]
    y = vals[:,-len(prefs):]
    # split into training and test sets
    ind_split = int(val_split * len(x))
    x_tr, x_ts = x[:-ind_split], x[-ind_split:]
    y_tr, y_ts = y[:-ind_split], y[-ind_split:]

    return (x_tr,y_tr), (x_ts,y_ts), features, prefs


if __name__ == '__main__':

    dem = get_dem_data('DEMO_J.XPT')
    ##dbq = get_dbq_data('DBQ_J.XPT')
    ##(x_tr,y_tr), (x_ts,y_ts), features, prefs = make_train_data(dem, dbq)
    alq = get_alq_data('ALQ_J.XPT')
    (x_tr,y_tr), (x_ts,y_ts), features, prefs = make_train_data(dem, alq)

