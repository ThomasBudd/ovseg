import torch
import torch.nn.functional as F
import numpy as np
import os
from ovseg.utils.io import load_pkl
from ovseg.networks.resUNet import UNetResEncoder
from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction

FLIP_AND_ROTATE_IMAGE = True


def ClaraWrapperOvarian(data_tpl,
                        models,
                        path_to_clara_models='/aiaa_workspace/aiaa-1/lib/ovseg_zxy/clara_models'):
    '''
    General wrapper for HGSOC segmentation.
    Can run the segmentation for different and multiple locations

    Parameters
    ----------
    data_tpl : dict
        contains 'image', 3D or 4D np array z first, and 'spacing' of len 3
    models : str or list of strings
        name of the folder in which the model parameters and weights are stored
    path_to_clara_models : str, optional
        location of the models on the server. The default is '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/clara_models'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if isinstance(models, str):
        
        models = [models]
    
    pred = evaluate_segmentation_ensemble(data_tpl,
                                          models[0],
                                          path_to_clara_models)
    
    torch.cuda.empty_cache()
    
    for model in models[1:]:
        
        arr = evaluate_segmentation_ensemble(data_tpl,
                                             model,
                                             path_to_clara_models)
        
        # fill in new prediction and overwrite previous one
        pred = pred * (arr == 0).type(torch.float) + arr
        torch.cuda.empty_cache()
        
    # back to numpy
    pred = pred.cpu().numpy()
    
    # put z axis back
    # numpy does this faster than torch
    return np.moveaxis(pred, 0, -1)

# %%
def preprocess_dynamic_z_spacing(data_tpl,
                                 prep_params):
    '''
    This function implements the dynamic z resizing used during inference.
    When given an image with a low slice thickness, it is resized to multiple
    images with non overlapping slices of target spacing to provide
    high resolution predictions.

    Parameters
    ----------
    data_tpl : dict
        Need to contain 'image', 3D or 4D np.ndarray and 'spacing' of len 3.
    prep_params : dict
        preprocessing parameters as used for the network training

    Returns
    -------
    im_list : list
        contains 5D torch cuda tensors of all images
    '''

    print('*** PREPROCESSING ***')
    im = data_tpl['image']
    if len(im.shape) == 3:
        im = im[np.newaxis]

    rif = data_tpl['raw_image_file']
    is_nifti = rif.endswith('.nii.gz') or rif.endswith('.nii')

    if FLIP_AND_ROTATE_IMAGE and is_nifti:
        # this corrects for differences in how dcms are read in ov_seg
        # and how clara creates nifti files from dcms
        im = np.rot90(im[:, ::-1, :, ::-1], -1, (1,2))

    # now the image should be 5d
    im = torch.from_numpy(im.copy()).type(torch.float).unsqueeze(0).cuda()
        
    # %% resizing, the funny part
    z_sp = data_tpl['spacing'][0]
    
    target_spacing = prep_params['target_spacing']
    target_z_spacing = target_spacing[0]
    
    # %% dynamic z spacing
    n_ims = int(np.max([np.floor(target_z_spacing/z_sp), 1]))
    print(f'Creating {n_ims} images with z spacing {target_z_spacing}')
    dynamic_z_spacing = target_z_spacing / n_ims
    
    scale_factor = [data_tpl['spacing'][0] / dynamic_z_spacing,
                    data_tpl['spacing'][1] / target_spacing[1],
                    data_tpl['spacing'][2] / target_spacing[2]]
    
    # resizing
    im = F.interpolate(im,
                       scale_factor=scale_factor,
                       mode='trilinear')
    
    # apply windowing            
    if prep_params['apply_windowing']:
        im = im.clamp(*prep_params['window'])
    
    # now rescaling
    scaling = prep_params['scaling']
    im = (im - scaling[1]) / scaling[0]
    
    # split images
    im_list = [im[:, :, i::n_ims] for i in range(n_ims)]
    
    # %% finally pooling
    if prep_params['apply_pooling']:
        stride = prep_params['pooling_stride']
        im_list = [F.avg_pool3d(im, kernel_size=stride, stride=stride) for im in im_list]
    
    # remove batch dimension
    im_list = [im[0] for im in im_list]
    
    return im_list

# %%
def evaluate_segmentation_ensemble(data_tpl,
                                   model,
                                   path_to_clara_models='/aiaa_workspace/aiaa-1/lib/ovseg_zxy/clara_models'):

    print(f'*** EVALUATING {model} ***')
    # At this path the model parameters and networks weights should be
    # stored
    path_to_model = os.path.join(path_to_clara_models, model)
    # Read model parameters
    path_to_model_params = os.path.join(path_to_model, 'model_parameters.pkl')
    model_params = load_pkl(path_to_model_params)

    im_list = preprocess_dynamic_z_spacing(data_tpl,
                                           model_params['preprocessing'])
    
    # dimensions of target tensor
    nz = np.sum([im.shape[1] for im in im_list])
    nx, ny = im_list[0].shape[2], im_list[0].shape[3]
    
    print('*** RUNNING THE MODEL ***')
    print('the fun starts...')

    # this needs updating to allow general architecture
    if not model_params['architecture'] == 'unetresencoder':
        raise NotImplementedError('Only implemented for ResEncoder so far...')
    
    network = UNetResEncoder(**model_params['network']).cuda()
    
    n_ch = model_params['network']['out_channels']
    
    # %% Sliding window prediction time!
    prediction = SlidingWindowPrediction(network=network,
                                         **model_params['prediction'])
    
    # collect all weights we got from the ensemble
    weight_files = [os.path.join(path_to_model, file) for file in os.listdir(path_to_model)
                    if file.startswith('network_weights')]
    
    # list of predictions from each weight
    pred_list = []
    # iterate over all weights used in the ensemble
    for j, weight_file in enumerate(weight_files):
        print(f'Evaluate network {j+1} out of {len(weight_files)}')
        # load weights
        prediction.network.load_state_dict(torch.load(weight_file,
                                                      map_location=torch.device('cuda')))
        
        # full tensor of softmax outputs
        # pred = torch.zeros((n_ch, nz, nx, ny), device='cuda', dtype=torch.float)
        # we're using numpy arrays here to prevent OOM errors
        pred = np.zeros((n_ch, nz, nx, ny), dtype=np.float32)
        
        # for each image in the list, evaluate sliding window and fill in
        for i, im in enumerate(im_list):            
            pred[:, i::len(im_list)] = prediction(im).detach().cpu().numpy()
        
        pred_list.append(pred)
    
    # this solution is ugly, but otherwise there might be OOM errors
    pred = np.stack(pred_list).mean(0)
    pred = torch.from_numpy(pred).cuda()
    torch.cuda.empty_cache()
    # %% we do the postprocessing manually here to save some moving to the
    # GPU back and fourth
    print('*** POSTPROCESSING ***')
    if 'postprocessing' in model_params:
        
        print('WARNING: Only resizing and argmax is performed here')
    
    # first trilinear resizing
    size = [int(s) for s in data_tpl['image'].shape[-3:]]
    
    try:
        pred = F.interpolate(pred.unsqueeze(0),
                             size=size,
                             mode='trilinear')[0]
    except RuntimeError:
        print('Went out of memory. Resizing again on the CPU, but this can be slow...')
        
        pred = F.interpolate(pred.unsqueeze(0).cpu(),
                             size=size,
                             mode='trilinear')[0]
        

    # now applying argmax
    pred = torch.argmax(pred, 0).type(torch.float)
    
    # now convert labels back to their orig. classes
    pred_lb = torch.zeros_like(pred)
    for i, lb in enumerate(model_params['preprocessing']['lb_classes']):
        # this should be the fastest way on the GPU to get the job done
        pred_lb = pred_lb + lb * (pred == i+1).type(torch.float)
    
    return pred_lb