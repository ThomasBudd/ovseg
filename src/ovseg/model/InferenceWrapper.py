import torch
import torch.nn.functional as F
import numpy as np
import os
from ovseg import OV_DATA_BASE
from ovseg.utils.io import load_pkl
from ovseg.networks.resUNet import UNetResEncoder
from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm



def preprocess(im, spacing, prep_params, use_dynamic_z_spacing=True):
    '''
    This function implements the dynamic z resizing used during inference.
    When given an image with a low slice thickness, it is resized to multiple
    images with non overlapping slices of target spacing to provide
    high resolution predictions.

    Parameters
    ----------
    im : np.ndarray
        3D or 4D array representing the 3d volume (ch, z, x, y)
    spacing : np.ndarray
        len 3, voxel spacing in [mm]
    prep_params : dict
        preprocessing parameters as used for the network training
    use_dynamic_z_spacing : bool
        splits images with low slice distance into multiple images to
        reduce interpolation artefacts on the z axis

    Returns
    -------
    im_list : list
        contains 5D torch cuda tensors of all images
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('*** PREPROCESSING ***')
    if len(im.shape) == 3:
        im = im[np.newaxis]
    # to 5d torch tensor
    im = torch.from_numpy(im).type(torch.float).unsqueeze(0).to(device)
        
    z_sp = spacing[0]
    
    target_spacing = prep_params['target_spacing']
    target_z_spacing = target_spacing[0]
    
    if use_dynamic_z_spacing:
        n_ims = int(np.max([np.floor(target_z_spacing/z_sp), 1]))
    else:
        n_ims = 1
    print(f'Creating {n_ims} image(s) with z spacing {target_z_spacing}')
    dynamic_z_spacing = target_z_spacing / n_ims
    
    scale_factor = [spacing[0] / dynamic_z_spacing,
                    spacing[1] / target_spacing[1],
                    spacing[2] / target_spacing[2]]
    
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
    
    # split images and remove batch dim
    im_list = [im[0, :, i::n_ims] for i in range(n_ims)]
    
    if len(im_list) == 1:
        print(f"Shape after preprocessing: {tuple(im_list[0].shape)}")
    else:
        print("Shapes after preprocessing:")
        for im in im_list:
            print(tuple(im.shape))
    
    return im_list

flip_dims = [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]

def sliding_window(volume, 
                   network, 
                   dev,
                   patch_size,
                   sigma_gaussian_weight=1/8,
                   overlap=0.5,
                   batch_size=1,
                   use_TTA=True, 
                   **kwargs):
    
    patch_size = np.array(patch_size)
    
    # thanks to Fabian Isensee! I took this from his code:
    # https://github.com/MIC-DKFZ/nnUNet/blob/14992342919e63e4916c038b6dc2b050e2c62e3c/nnunet/network_architecture/neural_network.py#L250
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_gaussian_weight for i in patch_size]
    tmp[tuple(center_coords)] = 1
    patch_weight = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    patch_weight = patch_weight / np.max(patch_weight) * 1
    patch_weight = patch_weight.astype(np.float32)

    # patch_weight cannot be 0, otherwise we may end up with nans!
    patch_weight[patch_weight == 0] = np.min(patch_weight[patch_weight != 0])
    patch_weight = torch.from_numpy(patch_weight).to(dev).float()
    
    # pad the image in case it is smaller than the patch size
    shape_in = np.array(volume.shape)
    pad = [0, patch_size[2] - shape_in[3], 0, patch_size[1] - shape_in[2],
           0, patch_size[0] - shape_in[1]]
    pad = np.maximum(pad, 0).tolist()
    volume = F.pad(volume, pad).type(torch.float)
    shape = volume.shape[1:]
    
    pred = torch.zeros((network.out_channels, *shape),
                       device=dev,
                       dtype=torch.float)
    # this is for the voxel where we have no prediction in the end
    # for each of those the method will return the (1,0,..,0) vector
    # pred[0] = 1
    ovlp = torch.zeros((1, *shape),
                       device=dev,
                       dtype=torch.float)

    # compute list of cropping cooridnates
    nz, nx, ny = shape

    n_patches = np.ceil((np.array([nz, nx, ny]) - patch_size) / 
                        (overlap * patch_size)).astype(int) + 1

    # upper left corners of all patches
    z_list = np.linspace(0, nz - patch_size[0], n_patches[0]).astype(int).tolist()
    x_list = np.linspace(0, nx - patch_size[1], n_patches[1]).astype(int).tolist()
    y_list = np.linspace(0, ny - patch_size[2], n_patches[2]).astype(int).tolist()

    zxy_list = []
    for z in z_list:
        for x in x_list:
            for y in y_list:
                zxy_list.append((z, x, y))
    
    n_full_batches = len(zxy_list) // batch_size
    zxy_batched = [zxy_list[i * batch_size: (i + 1) * batch_size]
                   for i in range(n_full_batches)]

    if n_full_batches * batch_size < len(zxy_list):
        zxy_batched.append(zxy_list[n_full_batches * batch_size:])

    # %% now the magic!
    with torch.no_grad():
        for zxy_batch in tqdm(zxy_batched):
            # crop
            batch = torch.stack([volume[:,
                                        z:z+patch_size[0],
                                        x:x+patch_size[1],
                                        y:y+patch_size[2]] for z, x, y in zxy_batch])

            # remember that the network is outputting a list of predictions for each scale
            if dev.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    out = network(batch)[0]
            else:
                out = network(batch)[0]

            if use_TTA:
                for dims in flip_dims[1:]:
                    batch_flp = torch.flip(batch, dims)
                    if dev.startswith('cuda'):
                        with torch.cuda.amp.autocast():
                            out_flp = network(batch_flp)[0]
                    else:
                        out_flp = network(batch_flp)[0]
                    
                    out += torch.flip(out_flp, dims)
                
                out /= len(flip_dims)

            # update pred and overlap
            for i, (z, x, y) in enumerate(zxy_batch):
                pred[:, z:z+patch_size[0], x:x+patch_size[1],
                     y:y+patch_size[2]] += F.softmax(out[i], 0) * patch_weight
                ovlp[:, z:z+patch_size[0], x:x+patch_size[1],
                     y:y+patch_size[2]] += patch_weight

        # %% bring maybe back to old shape
        pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
        ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

        # set the prediction to background and prevent zero division where
        # we did not evaluate the network
        pred[0, ovlp[0] == 0] = 1
        ovlp[ovlp == 0] = 1

        pred /= ovlp

        # just to be sure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred.cpu().numpy()
    

def evaluate_segmentation_model(im,
                                spacing,
                                model,
                                fast=False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'*** EVALUATING {model} ***')
    # At this path the model parameters and networks weights should be
    # stored
    path_to_model = os.path.join(OV_DATA_BASE, 'clara_models', model)
    # Read model parameters
    path_to_model_params = os.path.join(path_to_model, 'model_parameters.pkl')
    model_params = load_pkl(path_to_model_params)

    # store for later
    orig_shape = im.shape[-3:]

    im_list = preprocess(im, spacing, model_params['preprocessing'], not fast)
    
    # dimensions of target tensor
    nz = np.sum([im.shape[1] for im in im_list])
    nx, ny = im_list[0].shape[2], im_list[0].shape[3]
    
    print('*** RUNNING THE MODEL ***')
    print('the fun starts...')

    # this needs updating to allow general architecture
    if not model_params['architecture'] == 'unetresencoder':
        raise NotImplementedError('Only implemented for ResEncoder so far...')
    
    network = UNetResEncoder(**model_params['network']).to(device)
    
    n_ch = model_params['network']['out_channels']
    
    # %% Sliding window prediction time!
    # mode = 'simple' if fast else 'flip' # we disable TTA flipping in fast mode
    # prediction = SlidingWindowPrediction(network=network,
    #                                      **model_params['prediction'])
    
    pred_params = model_params['prediction']
    pred_params['use_TTA'] = not fast
    
    # collect all weights we got from the ensemble
    weight_files = [os.path.join(path_to_model, file) for file in sorted(os.listdir(path_to_model))
                    if file.startswith('network_weights')]
    if fast:
        # we only use one network in fast mode
        weight_files = weight_files[:1]
    
    # list of predictions from each weight
    pred_list = []
    # iterate over all weights used in the ensemble
    for j, weight_file in enumerate(weight_files):
        print(f'Evaluate network {j+1} out of {len(weight_files)}')
        # load weights
        network.load_state_dict(torch.load(weight_file,
                                           map_location=device))
        
        # full tensor of softmax outputs
        # pred = torch.zeros((n_ch, nz, nx, ny), device='cuda', dtype=torch.float)
        # we're using numpy arrays here to prevent OOM errors
        pred = np.zeros((n_ch, nz, nx, ny), dtype=np.float32)
        
        # for each image in the list, evaluate sliding window and fill in
        for i, im in enumerate(im_list):            
            pred[:, i::len(im_list)] = sliding_window(im, 
                                                      network, 
                                                      device, 
                                                      **pred_params)
        
        pred_list.append(pred)
    
    # this solution is ugly, but otherwise there might be OOM errors
    pred = np.stack(pred_list).mean(0)
    pred = torch.from_numpy(pred).to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # %% we do the postprocessing manually here to save some moving to the
    # GPU back and fourth
    print('*** POSTPROCESSING ***')
    if 'postprocessing' in model_params:
        
        print('WARNING: Only resizing and argmax is performed here')
    
    # first trilinear resizing
    size = [int(s) for s in orig_shape]
    
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

def InferenceWrapper(im, spacing, models, fast=False):
    
    if not torch.cuda.is_available():
        if not fast:
            print('WARNING: No GPU found, inference can be very slow, '
                  'consider changing to fast mode by adding the \"--fast\" to '
                  'your python call.')
        else:
            print('WARNING: No GPU found, inference can be slow.')
    
    if isinstance(models, str):
        
        models = [models]
    
    pred = evaluate_segmentation_model(im,
                                       spacing,
                                       models[0],
                                       fast=fast)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for model in models[1:]:
        
        arr = evaluate_segmentation_model(im,
                                          spacing,
                                          model,
                                          fast=fast)
        
        # fill in new prediction and overwrite previous one
        pred = pred * (arr == 0).type(torch.float) + arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    # back to numpy
    pred = pred.cpu().numpy()
    
    return pred
