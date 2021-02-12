import numpy as np
import torch
from os.path import join, basename, exists
from os import makedirs, environ
from ovseg.utils.io import save_nii
import pickle
import matplotlib.pyplot as plt
from ovseg.networks.recon_networks import reconstruction_network_fbp_convs, get_operator, \
    learned_primal_dual
from ovseg.training.ReconstructionNetworkTraining import \
    ReconstructionNetworkTraining
from ovseg.data.ReconstructionData import ReconstructionData
from tqdm import tqdm
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type
from ovseg.preprocessing.Reconstruction2dSimPreprocessing import \
    Reconstruction2dSimPreprocessing


class Reconstruction2dSimModel(ModelBase):

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', batch_size_val=4,
                 fp32_val=False, plot_window=[-150, 250]):
        self.dev = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        super().__init__(val_fold=val_fold, data_name=data_name,
                         model_name=model_name,
                         model_parameters=model_parameters,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name,
                         is_inference_only=is_inference_only,
                         fmt_write=fmt_write)
        self.batch_size_val = batch_size_val
        self.fp32_val = fp32_val
        self.plot_window = plot_window

    def initialise_preprocessing(self):
        operator_kwargs = self.model_parameters['operator']
        self.operator = get_operator(**operator_kwargs)
        preprocessing_kwargs = self.model_parameters['preprocessing']
        self.preprocessing = Reconstruction2dSimPreprocessing(self.operator,
                                                              **preprocessing_kwargs)
        print('preprocessing initialised')

    def initialise_augmentation(self):
        print('no augmentation needed')

    def initialise_network(self):
        architecture = self.model_parameters['architecture'].lower()
        if architecture == 'reconstruction_network_fbp_convs':
            self.network = reconstruction_network_fbp_convs(self.operator)
        elif architecture in ['lpd', 'learnedprimaldual', 'learned-primal-dual']:
            self.network = learned_primal_dual(self.operator)
        self.network = self.network.to(self.dev)

    def initialise_postprocessing(self):
        self.mu_water = self.preprocessing.mu_water
        self.window = self.preprocessing.window
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
        mm = [0] if self.window is None else [0, 1]
        if isinstance(im, np.ndarray):
            im = im.clip(*mm)
        elif torch.is_tensor(im):
            im = im.clamp(*mm)
        else:
            raise TypeError('Input must be np.ndarray or torch tensor')
        if self.preprocessing.window is None:
            im_HU = (im - self.mu_water)/self.mu_water*1000
        else:
            im_HU = im * (self.window[1] - self.window[0]) + self.window[0]

        return im_HU

    def predict(self, data_tpl, return_torch=False):
        '''
        predict(proj)

        evaluates the projection data of a full scan and return the 3d image

        '''
        if 'projection' in data_tpl:
            proj = data_tpl['projection']
        else:
            proj, _ = self.preprocessing.preprocess_volume(self, data_tpl['image'])
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

    def __call__(self, data_tpl, return_torch=False):
        return self.predict(data_tpl, return_torch)

    def fbp(self, proj):

        proj = torch.tensor(proj).type(torch.float)
        shape = np.array(proj.shape)
        if len(shape) == 2:
            proj = proj.reshape(1, 1, *shape)
        elif len(shape) == 3:
            proj = torch.stack([proj[..., z] for z in range(shape[-1])]).unsqueeze(1)
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

    def save_prediction(self, data_tpl, ds_name, filename=None):
        # if not file name is given
        if filename is None:
            filename = basename(data_tpl['raw_label_file'])

        # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
        pred_folder = join(environ['OV_DATA_BASE'], 'predictions', self.data_name,
                           self.model_name, ds_name+'_{}'.format(self.val_fold))
        if not exists(pred_folder):
            makedirs(pred_folder)

        pred = data_tpl[self.pred_key]
        spacing = data_tpl['spacing']
        save_nii(pred, join(pred_folder, filename), spacing)

    def plot_prediction(self, data_tpl, ds_name, filename=None):
        # find name of the file
        if filename is None:
            filename = basename(data_tpl['raw_label_file'])

        # remove fileextension e.g. .nii.gz
        filename = filename.split('.')[0]

        # all predictions are stored in the designated 'plots' folder in the OV_DATA_BASE
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', self.data_name,
                           self.model_name, ds_name+'_{}'.format(self.val_fold))
        if not exists(plot_folder):
            makedirs(plot_folder)

        # get the data needed for plotting
        im = data_tpl['image']
        if 'projection' in data_tpl:
            # in this case the image is not in HU, let's convert
            im = self.postprocessing(im)
        im = im.clip(*self.plot_window)
        proj = data_tpl['projection']
        pred = data_tpl[self.pred_key]
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
        plt.savefig(join(plot_folder, filename+'.png'))
        plt.close(fig)

    def compute_error_metrics(self, data_tpl):
        pred = data_tpl[self.pred_key]
        im = data_tpl['image']
        if 'projection' in data_tpl:
            # in this case the image is not in HU, let's convert
            im = self.postprocessing(im)
        mse = np.mean((pred - im)**2)
        Imax2 = (im - im.min()).max()**2
        PSNR = 10 * np.log10(Imax2 / mse)
        im_win = im.clip(*self.plot_window)
        pred_win = pred.clip(*self.plot_window)
        mse_win = np.mean((pred_win - im_win)**2)
        Imax2_win = (im_win - im_win.min()).max()**2
        PSNR_win = 10 * np.log10(Imax2_win / mse_win)
        return {'PSNR': PSNR, 'PSNR_win': PSNR_win}

    def _init_global_metrics(self):
        self.global_metrics_helper = {'squared_error': 0, 'squared_error_win': 0,
                                      'Imax_squared': -np.inf, 'Imax_squared_win': -np.inf, 'n': 0}
        self.global_metrics = {'PSNR': -1, 'PSNR_win': -1}

    def _update_global_metrics(self, data_tpl):
        im = self.postprocessing(data_tpl['image'])
        pred = data_tpl[self.pred_key]
        im_win = im.clip(*self.plot_window)
        pred_win = pred.clip(*self.plot_window)

        # get the helper variables
        sq_err = self.global_metrics_helper['squared_error']
        sq_err_win = self.global_metrics_helper['squared_error_win']
        Imax_sq = self.global_metrics_helper['Imax_squared']
        Imax_sq_win = self.global_metrics_helper['Imax_squared_win']
        n = self.global_metrics_helper['n']

        # update helper variables
        sq_err += np.sum((im - pred)**2) / 10**7
        sq_err_win += np.sum((im_win - pred_win)**2) / 10**7
        n += np.prod(im.shape) / 10**7
        Imax_sq = max([Imax_sq, (im.max() - im.min())**2])
        Imax_sq_win = max([Imax_sq, (im_win.max() - im_win.min())**2])

        # now update the metrics
        self.global_metrics['PSNR'] = 10 * np.log10(Imax_sq / sq_err)
        self.global_metrics['PSNR_win'] = 10 * np.log10(Imax_sq_win / sq_err_win)

        # and store the helpers again
        self.global_metrics_helper['squared_error'] = sq_err
        self.global_metrics_helper['squared_error_win'] = sq_err_win
        self.global_metrics_helper['Imax_squared'] = Imax_sq
        self.global_metrics_helper['Imax_squared_win'] = Imax_sq_win
        self.global_metrics_helper['n'] = n
