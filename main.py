'''
    this is a prototype code for the 'Behavioral ML' project
'''

from data_processing import load_data, split_data
from models import setup_model, train_model_ml, train_model_bs
from besci import besci_loss
from visualization import plot_losses, plot_metrics, plot_overfit, plot_dem_sensitivity


def run_experiment(params):
    '''run a single experiment'''
    # get train/test data
    data = load_data(params['data_file'])
    ##(x_tr,y_tr), (x_ts,y_ts) = split_data(data, val_split=params['val_split'], seed=params['seed'])
    ##input_size, output_size = x_tr.shape[1], y_tr.shape[1]
    train_data, test_data = split_data(data, val_split=params['val_split'], seed=params['seed'])
    input_size, output_size = train_data[0].shape[1], train_data[1].shape[1]

    # create ml and bs models
    models = {'ml': setup_model(input_size, output_size, arch=params['arch'], seed=params['seed']),
              'bs': setup_model(input_size, output_size, arch=params['arch'], seed=params['seed'])}

    # train ml and bs models
    res = {}
    ml_loss, ml_mse, ml_ent = train_model_ml(models['ml'], params, train_data, test_data)
    bs_loss, bs_mse, bs_ent = train_model_bs(models['bs'], params, train_data, test_data)
    res = {'loss': {'ml': ml_loss, 'bs': bs_loss},
           'mse': {'ml': ml_mse, 'bs': bs_mse},
           'ent': {'ml': ml_ent, 'bs': bs_ent}}

    return train_data, test_data, models, res


def visualize_experiment(models, res, test_data):
    '''visualize results of an experiment'''
    # plot loss and metrics
    plot_losses(res['loss'], name='losses')
    plot_metrics(res['mse'], name='mse')
    plot_metrics(res['ent'], name='ent')
    plot_overfit(res['mse'], name='mse')
    plot_overfit(res['ent'], name='ent')

    # plot sensitivity to demographic features
    dem_features = ['age', 'income', 'gender_male', 'education']
    for dem_ind in [0,1,2,3]:
        plot_dem_sensitivity(models, test_data, dem_ind, name=dem_features[dem_ind])


if __name__ == '__main__':

    # specify experiment parameters
    params = {'data_file': './data/wain_dataset_one_hot.csv',
              'seed': 2022, 'val_split': 0.1,
              'arch': [32,32], 'batch_size': 32, 'num_epoch': 100}

    # run and visualize an experiment
    train_data, test_data, models, res = run_experiment(params)
    visualize_experiment(models, res, test_data)

