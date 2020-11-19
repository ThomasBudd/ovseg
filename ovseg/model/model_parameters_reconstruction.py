

def get_model_params_2d_reconstruction(dose, batch_size=12,
                                       image_folder='images'):
    model_parameters = {}

    trn_dl_params = {'batch_size': batch_size, 'epoch_len': 250}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    keys = ['image', 'projection', 'spacing']
    folders = [image_folder, 'projections_'+dose, 'spacings']
    data_params = {'n_folds': 5, 'fixed_shuffle': True,
                   'trn_dl_params': trn_dl_params,
                   'val_dl_params': val_dl_params,
                   'keys': keys, 'folders': folders}
    model_parameters['data'] = data_params

    # now finally the training!
    loss_params = {'l1weight': 0}
    opt_params = {'lr': 10**-4}
    lr_params = {'beta': 0.9, 'lr_min': 10**-6}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 1000, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7, 'fp32': False,
                       'p_plot_list': [0, 0.5, 0.8], 'opt_name': 'ADAM'}
    model_parameters['training'] = training_params
    return model_parameters
