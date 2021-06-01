

def get_model_params_2d_restauration(image_folder='images_restauration',
                                     fbp_folder='fbps',
                                     fp32=False):
    model_parameters = {}
    trn_dl_params = {'batch_size': 4,
                     'epoch_len': 250,
                     'image_key': 'image',
                     'fbp_key': 'fbp',
                     'store_data_in_ram': False,
                     'return_fp16': True,
                     'n_max_volumes': None,
                     'n_bias': 0}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    val_dl_params['store_data_in_ram'] = True
    val_dl_params['n_max_volumes'] = 50
    keys = ['image', 'fbp']
    folders = [image_folder, fbp_folder]
    data_params = {'n_folds': 5, 'fixed_shuffle': True,
                   'trn_dl_params': trn_dl_params,
                   'val_dl_params': val_dl_params,
                   'keys': keys, 'folders': folders}
    model_parameters['data'] = data_params
    network_params={'in_channels': 1, 'out_channels': 1, 'filters': 32, 'filters_max': 320,
                    'n_stages': 5, 'conv_params': None, 'norm': 'inst', 'norm_params': None,
                    'nonlin_params': None, 'double_channels_every_second_block': True}
    model_parameters['network'] = network_params
    # now finally the training!
    opt_params = {'lr': 10**-4}
    lr_schedule='lin_ascent_cos_decay'
    lr_params = {'n_warmup_epochs': 50, 'lr_max': 3 * 10**-4}
    training_params = {'num_epochs': 1000, 'opt_params': opt_params,
                       'opt_name': 'ADAM', 'fp32': fp32, 'lr_schedule': lr_schedule,
                       'lr_params': lr_params}
    model_parameters['training'] = training_params
    model_parameters['prediction_key'] = 'learned_restauration'
    return model_parameters
