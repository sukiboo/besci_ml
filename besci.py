'''
    here we define the behavioral loss function that is used by 'Behavioral ML'

    !!! don't forget to change it if we change our data encoding !!!

'''

import numpy as np
import tensorflow as tf


def uncertainty_age(age):
    '''estimate uncertainty based on age'''
    return .5*age


def uncertainty_income(income):
    '''estimate uncertainty based on income'''
    return 0


def uncertainty_gender(gender):
    '''estimate uncertainty based on gender'''
    return 0


def uncertainty_race(race):
    '''estimate uncertainty based on race'''
    return 0


def uncertainty_edu(edu):
    '''estimate uncertainty based on education'''
    return 0


def uncertainty_marital(marital):
    '''estimate uncertainty based on marital status'''
    return 0


def besci_loss(x, y, z):
    '''behavioral-science-informed loss function'''
    # extract demographic features -- they are hardcoded and this is bad
    d_age, d_income, d_gender, d_race, d_edu, d_marital = np.split(x, [1,2,3,8,12], axis=1)

    # compute uncertainties based on demographic features
    u_age = uncertainty_age(d_age)
    u_income = uncertainty_income(d_income)
    u_gender = uncertainty_gender(d_gender)
    u_race = uncertainty_race(d_race)
    u_edu = uncertainty_edu(d_edu)
    u_marital = uncertainty_marital(d_marital)

    # compute besci-loss
    u = 1 - u_age * u_income * u_gender * u_race * u_edu * u_marital
    loss = tf.reduce_mean(-u * tf.reduce_sum(y * tf.math.log(z), axis=1))

    return loss

