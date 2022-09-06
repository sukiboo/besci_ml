'''
    this is a prototype code for the 'Behavioral ML' project
'''

from data_processing import load_data, split_data
from models import setup_model, train_model_ml, train_model_bs
from besci import besci_loss
from visualization import plot_losses, plot_metrics, plot_overfit, plot_dem_sensitivity


if __name__ == '__main__':

    # specify experiment parameters
    params = {'data_file': './data/wain_dataset_one_hot.csv',
              'seed': 2022, 'val_split': 0.1,
              'arch': [32,32], 'batch_size': 32, 'num_epoch': 100}

    # get train/test data
    data = load_data(params['data_file'])
    (x_tr,y_tr), (x_ts,y_ts) = split_data(data, val_split=params['val_split'], seed=params['seed'])
    input_size, output_size = x_tr.shape[1], y_tr.shape[1]

    # create ml and bs models
    model_ml = setup_model(input_size, output_size, arch=params['arch'], seed=params['seed'])
    model_bs = setup_model(input_size, output_size, arch=params['arch'], seed=params['seed'])

    # train ml and bs models
    ml_loss, ml_mse, ml_ent = train_model_ml(model_ml, params, (x_tr,y_tr), (x_ts,y_ts))
    bs_loss, bs_mse, bs_ent = train_model_bs(model_bs, params, (x_tr,y_tr), (x_ts,y_ts))

    # plot loss and metrics
    plot_losses([ml_loss[0], bs_loss[0]], [ml_loss[1], bs_loss[1]], name='losses')
    plot_metrics([ml_mse[0], bs_mse[0]], [ml_mse[1], bs_mse[1]], name='mse')
    plot_metrics([ml_ent[0], bs_ent[0]], [ml_ent[1], bs_ent[1]], name='ent')
    plot_overfit(ml_mse, bs_mse, name='mse')
    plot_overfit(ml_ent, bs_ent, name='ent')

    # plot sensitivity to demographic features
    for dem_ind in [0,1,2,3]:
        plot_dem_sensitivity(model_ml, model_bs, x_ts, dem_ind, name=data.columns[dem_ind])

