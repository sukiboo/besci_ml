
import numpy as np
import pandas as pd


def load_data(demographics_file, preferences_file):
    '''load and merge demographic and food preference data'''
    demographics = pd.read_sas(f'./data/{demographics_file}')
    preferences = pd.read_sas(f'./data/{preferences_file}')

    # merge demographics and food preferences
    df = demographics.merge(preferences, on='SEQN')
    columns = {'age': 'RIDAGEYR',
               'gender': 'RIAGENDR',
               'income': 'INDFMPIR',
               'education': 'DMDEDUC2',
               'race': 'RIDRETH1',
               'marital': 'DMDMARTL',
               'num_not_home_7': 'DBD895',
               'num_fast_food_7': 'DBD900',
               'num_ready_30': 'DBD905',
               'num_frozen_30': 'DBD910'}
    df = df[columns.values()]
    df.columns = columns.keys()

    # remove false values
    df = df.dropna()
    df = df.astype({'age': int,
                    'gender': int,
                    'income': float,
                    'education': int,
                    'race': int,
                    'marital': int,
                    'num_not_home_7': int,
                    'num_fast_food_7': int,
                    'num_ready_30': int,
                    'num_frozen_30': int})
    df = df[df.age.ge(20) & df.age.le(80) &
            df.gender.isin([1,2]) &
            df.income.ge(0.) & df.income.le(5.) &
            df.education.isin([1,2,3,4,5]) &
            df.race.isin([1,2,3,4,5,6,7]) &
            df.marital.isin([1,2,3,4,5,6])]
    df = df[df.num_not_home_7.ge(0) & df.num_not_home_7.le(21) &
            df.num_fast_food_7.ge(0) & df.num_fast_food_7.le(21) &
            df.num_not_home_7.ge(df.num_fast_food_7) &
            df.num_ready_30.ge(0) & df.num_ready_30.le(90) &
            df.num_frozen_30.ge(0) & df.num_frozen_30.le(90) &
            (df.num_ready_30 + df.num_frozen_30).le(90)]

    return df.reset_index(drop=True)


def encode_data(df):
    '''preprocess and encode the data'''

    # encode demographic features
    df.age = (df.age - 20) / 60
    df.gender = 2 - df.gender
    df.income = df.income / 5
    df.education = (df.education - 1) / 4
    race_dict = {1: .66, 2: .66, 3: 1., 4: .33, 5: 0.}
    marital_dict = {1: 1., 2: 0., 3: 0., 4: 0., 5: 0., 6: 1.}
    df = df.replace({'race': race_dict, 'marital': marital_dict})

    # encode food preferences as probability
    df['eat_out'] = (df.num_not_home_7 - df.num_fast_food_7) / 21
    df['fast_food'] = df.num_fast_food_7 / 21
    df['ready'] = df.num_ready_30 / 90
    df['frozen'] = df.num_frozen_30 / 90
    df['cook'] = 1 - df.eat_out - df.fast_food - df.ready - df.frozen
    df = df.drop(columns=['num_not_home_7', 'num_fast_food_7', 'num_ready_30', 'num_frozen_30'])

    return df, df.columns[:6], df.columns[6:]


if __name__ == '__main__':

    df = load_data('DEMO_J.XPT', 'DBQ_J.XPT')
    df, _, _ = encode_data(df)

