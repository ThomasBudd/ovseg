import numpy as np
import torch
from os.path import join, basename, exists
from os import mkdir
from ovseg.utils.io import save_nii
import pickle
import matplotlib.pyplot as plt
from ovseg.networks.recon_networks import learned_reconstruction_model, get_operator
from ovseg.training.ReconstructionNetworkTraining import \
    ReconstructionNetworkTraining
from ovseg.data.ReconstructionData import ReconstructionData
from tqdm import tqdm
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type


class ReconstructionModel(ModelBase):

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', mu=0.0192, batch_size_val=4,
                 fp32_val=False, plot_window=[-150, 250]):
        super().__init__(val_fold=val_fold, data_name=data_name,
                         model_name=model_name,
                         model_parameters=model_parameters,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name,
                         is_inference_only=is_inference_only,
                         fmt_write=fmt_write)
        self.mu = mu
        self.batch_size_val = batch_size_val
        self.fp32_val = fp32_val
        self.plot_window = plot_window

    def initialise_preprocessing(self):
        print('no preprocessing needed')

    def initialise_augmentation(self):
        print('no augmentation needed')

    def initialise_network(self):
        self.operator = get_operator()
        self.network = learned_reconstruction_model(self.operator)

    def initialise_postprocessing(self):
        print('postprocessing initialised')

    def initialise_data(self):
        print('initialise data')
        try:
            params = self.model_parameters['data']
        except KeyError:
            params = {}
            print('Warning! No data parameters found')

        self.data = ReconstructionData(self.val_fold, self.preprocessed_path,
                                       **params)

    def initialise_training(self):
        print('Initialise training')
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        params = self.model_parameters['training'].copy()

        self.training = \
            ReconstructionNetworkTraining(network=self.network,
                                          trn_dl=self.data.trn_dl,
                                          val_dl=self.data.val_dl,
                                          model_path=self.model_path,
                                          network_name=self.network_name,
                                          **params)

    def postprocessing(self, im):
        '''
        postprocessing(im)

        maps image gary values to HU

        '''
        if isinstance(im, np.ndarray):
            im = np.maximum(im, 0)
        elif torch.is_tensor(im):
            im = im.clip(0)
        else:
            raise TypeError('Input must be np.ndarray or torch tensor')
        im_HU = (im - self.mu)/self.mu*1000
        return im_HU

    def predict(self, data_dict, return_torch=False):
        '''
        predict(proj)

        evaluates the projection data of a full scan and return the 3d image

        '''
        proj = data_dict['projection']
        is_np, _ = check_type(proj)

        # make sure proj is torch tensor
        if is_np:
            proj = torch.from_numpy(proj).cuda()
        else:
            proj = proj.cuda()

        # add channel axes
        proj = proj.unsqueeze(0)

        nz = proj.shape[-1]
        bs = self.batch_size_val
        pred = torch.zeros((512, 512, nz), device='cuda')
        z_list = list(range(0, nz - bs, bs)) + [nz - bs]

        # do the iterations
        with torch.no_grad():
            for z in z_list:
                batch = torch.stack([proj[..., zb] for zb in range(z, z + bs)])
                if self.fp32_val:
                    out = self.network(batch)
                else:
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)
                # out back to gpu and reshape
                out = torch.stack([out[b, 0] for b in range(bs)], -1)
                pred[..., z: z + bs] = out

            # convert to HU
            pred = self.postprocessing(pred)

        if is_np and not return_torch:
            pred = pred.cpu().numpy()

        torch.cuda.empty_cache()

        return pred

    def fbp(self, proj):

        proj = torch.tensor(proj).type(torch.float)
        shape = np.array(proj.shape)
        if len(shape) == 2:
            proj = proj.reshape(1, 1, *shape)
        elif len(shape) == 3:
            proj = torch.stack([proj[..., z] for z in shape[-1]]).unsqueeze(1)
        elif not len(shape) == 4:
            raise ValueError('proj must be 2d projection, 3d stacked in last '
                             'dim or 4d in batch form.')

        # do the fbp
        with torch.no_grad():
            filtered_sinogram = self.operator.filter_sinogram(proj.to('cuda'))
            fbp = self.operator.backprojection(filtered_sinogram).cpu().numpy()

        # get the arrays in right shape again
        if len(shape) == 2:
            fbp = fbp[0, 0]
        elif len(shape == 3):
            # stack again in last dim
            fbp = np.moveaxis(fbp, 0, -1)[0]

        # if fbp is in batch from we don't have to do reshaping
        return fbp

    def save_prediction(self, pred, data, pred_folder, name):
        save_nii(pred, join(pred_folder, name + '.nii.gz'), data['spacing'])

    def plot_prediction(self, pred, data, plot_folder, name):
        im = self.postprocessing(data['image']).clip(*self.plot_window)
        proj = data['projection']
        # extract slices
        z = np.random.randint(im.shape[-1])
        im_sl = im[..., z]
        pred_sl = pred[..., z].clip(*self.plot_window)
        # compute fbp
        fbp_sl = self.fbp(proj[..., z])
        fbp_sl = self.postprocessing(fbp_sl).clip(*self.plot_window)
        # compute PSNR
        mse_fbp = np.mean((im_sl - fbp_sl) ** 2)
        mse_pred = np.mean((im_sl - pred_sl) ** 2)
        Imax2 = (im_sl - im_sl.min()).max()**2
        PSNR_pred = 10 * np.log10(Imax2 / mse_pred)
        PSNR_fbp = 10 * np.log10(Imax2 / mse_fbp)

        # now some plotting
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(fbp_sl, cmap='gray',
                   vmin=self.plot_window[0],
                   vmax=self.plot_window[1])
        plt.title('FBP: {:.2f}dB'.format(PSNR_fbp))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(im_sl, cmap='gray',
                   vmin=self.plot_window[0],
                   vmax=self.plot_window[1])
        plt.title('Target')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(pred_sl, cmap='gray',
                   vmin=self.plot_window[0],
                   vmax=self.plot_window[1])
        plt.title('CNN: {:.2f}dB'.format(PSNR_pred))
        plt.axis('off')
        plt.savefig(join(plot_folder, name+'.png'))
        plt.close(fig)

    def compute_error_metrics(self, pred, data):
        im = self.postprocessing(data['image'])
        mse = np.mean((pred - im)**2)
        Imax2 = (im - im.min()).max()**2
        PSNR = 10 * np.log10(Imax2 / mse)
        rel_mse = mse / np.mean(im**2)
        return {'PSNR': PSNR, 'rel_mse': rel_mse}
