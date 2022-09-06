
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='tab20', font='monospace')


def plot_losses(losses_tr, losses_ts, name=None):
    '''plot training and test losses'''
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(losses_tr[0], linewidth=3)
    plt.plot(losses_ts[0], linestyle='--', linewidth=3)
    plt.plot(losses_tr[1], linewidth=3)
    plt.plot(losses_ts[1], linestyle='--', linewidth=3)
    ##plt.yscale('log')
    plt.legend(['ml-training loss', 'ml-test loss', 'bs-training loss', 'bs-test loss'], loc='upper right')
    plt.title('Training and test losses')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/losses_{name}.png', dpi=300, format='png')
    plt.show()


def plot_metrics(metrics_tr, metrics_ts, name=None):
    '''plot training and test metrics'''
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(metrics_tr[0], linewidth=3)
    plt.plot(metrics_ts[0], linestyle='--', linewidth=3)
    plt.plot(metrics_tr[1], linewidth=3)
    plt.plot(metrics_ts[1], linestyle='--', linewidth=3)
    ##plt.yscale('log')
    plt.legend(['ml-training metric', 'ml-test metric', 'bs-training metric', 'bs-test metric'], loc='upper right')
    plt.title('Training and test metrics')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/metrics_{name}.png', dpi=300, format='png')
    plt.show()


def plot_overfit(ml_metrics, bs_metrics, name=None):
    '''plot ratio of training-to-test metrics'''
    ml_overfit = np.array(ml_metrics[1]) / np.array(ml_metrics[0])
    bs_overfit = np.array(bs_metrics[1]) / np.array(bs_metrics[0])
    sns.set_palette('tab10')
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(ml_overfit, linewidth=3)
    plt.plot(bs_overfit, linewidth=3)
    plt.ylim([.8,1.2])
    plt.legend(['ml-overfit', 'bs-overfit'], loc='upper right')
    plt.title('Overfit of ml and bs models')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/overfit_{name}.png', dpi=300, format='png')
    plt.show()


def plot_dem_sensitivity(model_ml, model_bs, x_ts, dem_ind, name=None):
    '''calculate and plot sensitivity to demographic features'''
    # calculate demographic sensitivity
    x = x_ts.copy()
    dists_ml, dists_bs = [], []
    for d in np.linspace(0,1,11):
        x[:,dem_ind] = d
        dists_ml.append(model_ml.predict(x).mean(axis=0))
        dists_bs.append(model_bs.predict(x).mean(axis=0))
    diffs_ml = [sum(abs(dists_ml[i+1] - dists_ml[i])) for i in range(len(dists_ml)-1)]
    diffs_bs = [sum(abs(dists_bs[i+1] - dists_bs[i])) for i in range(len(dists_bs)-1)]

    # plot demographic sensitivity
    sns.set_palette('tab10')
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(diffs_ml, linewidth=3)
    plt.plot(diffs_bs, linewidth=3)
    plt.ylim([.0,.05])
    plt.legend(['ml-sensitivity', 'bs-sensitivity'])
    if name is not None:
        plt.title(f'Sensitivity to {name}')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'./images/sensitivity_{name}.png', dpi=300, format='png')
    plt.show()

