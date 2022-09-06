'''
    here we define the behavioral loss function that is used by 'Behavioral ML'
'''

import numpy as np
import tensorflow as tf


def uncertainty_age(age):
    '''estimate uncertainty based on age'''
    # uncertainty decreases with age
    return .5 * (1 - age)


def uncertainty_income(income):
    '''estimate uncertainty based on income'''
    # uncertainty does not depend on income
    return 0


def uncertainty_gender(gender):
    '''estimate uncertainty based on gender'''
    # uncertainty is higher for males
    return .25 * gender


def uncertainty_race(race):
    '''estimate uncertainty based on race'''
    # uncertainty does not depend on race
    return 0


def uncertainty_edu(edu):
    '''estimate uncertainty based on education'''
    # uncertainty decreases with education level
    return .25 * (1 - edu)


def uncertainty_marital(marital):
    '''estimate uncertainty based on marital status'''
    # uncertainty is higher for married people
    return .5 * (1 - marital[:,0] - marital[:,1]).reshape(-1,1)


def besci_loss(x, y, z):
    '''behavioral-science-informed loss function'''
    # extract demographic features -- they are hardcoded and this is bad
    d_age, d_income, d_gender, d_edu, d_race, d_marital = np.split(x, [1,2,3,4,8], axis=1)

    # compute uncertainties based on demographic features
    u_age = uncertainty_age(d_age)
    u_income = uncertainty_income(d_income)
    u_gender = uncertainty_gender(d_gender)
    u_race = uncertainty_race(d_race)
    u_edu = uncertainty_edu(d_edu)
    u_marital = uncertainty_marital(d_marital)

    # compute besci-loss
    certainty = (1 - u_age) * (1 - u_income) * (1 - u_gender) * (1 - u_race) * (1 - u_edu) * (1 - u_marital)
    loss = tf.reduce_mean(certainty * tf.square(y - z))

    return loss

