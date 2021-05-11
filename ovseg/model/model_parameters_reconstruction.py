

def get_model_params_2d_reconstruction(architecture='reconstruction_network_fbp_convs',
                                       image_folder='images_recon',
                                       projection_folder='projections',
                                       fp32=False):
    model_parameters = {}
    batch_size = 4#12 if architecture == 'reconstruction_network_fbp_convs' else 4
    trn_dl_params = {'batch_size': batch_size, 'epoch_len': 250, 'store_data_in_ram': False,
                     'return_fp16': not fp32}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    val_dl_params['store_data_in_ram'] = True
    val_dl_params['n_max_volumes'] = 50
    keys = ['image', 'projection']
    folders = [image_folder, projection_folder]
    data_params = {'n_folds': 5, 'fixed_shuffle': True,
                   'trn_dl_params': trn_dl_params,
                   'val_dl_params': val_dl_params,
                   'keys': keys, 'folders': folders}
    model_parameters['data'] = data_params
    model_parameters['preprocessing'] = {'num_photons': 2*10**6, 'mu_water': 0.0192}
    model_parameters['architecture'] = architecture
    # now finally the training!
    loss_params = {'l1weight': 0}
    betas = (0.9, 0.999) if architecture not in ['LPD', 'lpd'] else (0.5, 0.99)
    lr = 10**-4 if architecture not in ['LPD', 'lpd'] else 5*10**-5
    opt_params = {'lr': lr, 'betas': betas}
    lr_params = {'beta': 0.9, 'lr_min': 0}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 300, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7,
                       'p_plot_list': [1, 0.5, 0.2], 'opt_name': 'ADAM', 'fp32': fp32}
    model_parameters['training'] = training_params
    model_parameters['prediction_key'] = 'learned_reconstruction'
    model_parameters['operator'] = {'n_angles': 256, 'det_count': 724}
    return model_parameters
