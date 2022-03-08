import numpy as np
import torch
from os.path import join, basename, exists
from os import makedirs, environ
from ovseg.utils.io import save_nii, load_pkl
import matplotlib.pyplot as plt
from ovseg.networks.restauration_networks import ResUNet2d
from ovseg.training.RestaurationNetworkTraining import \
    RestaurationNetworkTraining
from ovseg.data.RestaurationData import RestaurationData
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type
from ovseg.preprocessing.Restauration2dSimPreprocessing import \
    Restauration2dSimPreprocessing


class RestaurationModel(ModelBase):

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', batch_size_val=4,
                 fp32_val=False,
                 dont_store_data_in_ram=False,
                 preprocessed_parameters_name='restauration_parameters'):
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dont_store_data_in_ram = dont_store_data_in_ram
        self.preprocessed_parameters_name = preprocessed_parameters_name
        super().__init__(val_fold=val_fold, data_name=data_name,
                         model_name=model_name,
                         model_parameters=model_parameters,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name,
                         is_inference_only=is_inference_only,
                         fmt_write=fmt_write)
        self.batch_size_val = batch_size_val
        self.fp32_val = fp32_val

    def initialise_preprocessing(self):
        preprocessing_kwargs = load_pkl(join(environ['OV_DATA_BASE'], 'preprocessed',
                                             self.data_name, self.preprocessed_name, 
                                             self.preprocessed_parameters_name+'.pkl'))
        self.preprocessing = Restauration2dSimPreprocessing(**preprocessing_kwargs)
        self.model_parameters['preprocessing'] = preprocessing_kwargs
        if self.parameters_match_saved_ones:
            self.save_model_parameters()
        print('preprocessing initialised')

    def initialise_augmentation(self):
        print('no augmentation needed')

    def initialise_network(self):
        params = {} if 'network' not in self.model_parameters.keys() else \
            self.model_parameters['network']
        self.network = ResUNet2d(**params)
        self.network = self.network.to(self.dev)

    def initialise_postprocessing(self):
        self.mu_water = self.preprocessing.mu_water
        self.scaling = self.preprocessing.scaling
        self.window = self.preprocessing.window
        print('postprocessing initialised')

    def initialise_data(self):
        print('initialise data')
        try:
            params = self.model_parameters['data']
        except KeyError:
            params = {}
            print('Warning! No data parameters found')

        # if we don't want to store our data in ram...
        if self.dont_store_data_in_ram:
            for key in ['trn_dl_params', 'val_dl_params']:
                params[key]['store_data_in_ram'] = False
                # params[key]['store_coords_in_ram'] = False

        self.data = RestaurationData(self.val_fold, self.preprocessed_path, **params)

    def initialise_training(self):
        print('Initialise training')
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        params = self.model_parameters['training'].copy()

        self.training = RestaurationNetworkTraining(network=self.network,
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
        if self.scaling is not None:
            im = self.scaling[0] * im + self.scaling[1]
        if self.window is not None:
            im = im.clip(*self.window)
        else:
            im = im.clip(-1000)
        return im

    def predict(self, data_tpl, return_torch=False):
        '''
        predict(fbp)

        evaluates the fbpection data of a full scan and return the 3d image

        '''
        if 'fbp' in data_tpl:
            fbp = data_tpl['fbp']
        else:
            fbp, _ = self.preprocessing.preprocess_volume(data_tpl['image'])
        is_np, _ = check_type(fbp)

        # make sure fbp is torch tensor
        if is_np:
            fbp = torch.from_numpy(fbp).to(self.dev)
        else:
            fbp = fbp.to(self.dev)

        # add channel axes
        fbp = fbp.unsqueeze(1)

        nz = fbp.shape[0]
        bs = self.batch_size_val
        pred = torch.zeros((nz, 512, 512), device='cuda')
        z_list = list(range(0, nz - bs, bs)) + [nz - bs]

        # do the iterations
        with torch.no_grad():
            for z in z_list:
                batch = torch.stack([fbp[zb] for zb in range(z, z + bs)])
                if self.fp32_val:
                    out = self.network(batch)
                else:
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)
                # out back to gpu and reshape
                out = torch.stack([out[b, 0] for b in range(bs)])
                pred[z: z + bs] = out

            # convert to HU
            pred = self.postprocessing(pred)

        if is_np and not return_torch:
            pred = pred.cpu().numpy()

        torch.cuda.empty_cache()

        data_tpl[self.pred_key] = pred

        return pred

    def __call__(self, data_tpl, return_torch=False):
        return self.predict(data_tpl, return_torch)

    def save_prediction(self, data_tpl, folder_name, filename=None):
        # if not file name is given
        if filename is None:
            filename = basename(data_tpl['raw_image_file'])
            if filename.endswith('_0000.nii.gz'):
                filename = filename[:-12]

        # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
        pred_folder = join(environ['OV_DATA_BASE'], 'predictions', self.data_name,
                           self.model_name, folder_name)
        if not exists(pred_folder):
            makedirs(pred_folder)

        pred = data_tpl[self.pred_key]
        # in case of preprocessed data_tpls remember that the reconstruction is before
        # chaning of spacing
        spacing = data_tpl['orig_spacing'] if 'orig_spacing' in data_tpl else data_tpl['spacing']
        save_nii(pred, join(pred_folder, filename), spacing)

    def plot_prediction(self, data_tpl, folder_name, filename=None):
        # find name of the file
        if filename is None:
            filename = basename(data_tpl['raw_image_file'])
            if filename.endswith('_0000.nii.gz'):
                filename = filename[:-12]

        # remove fileextension e.g. .nii.gz
        filename = filename.split('.')[0]

        # all predictions are stored in the designated 'plots' folder in the OV_DATA_BASE
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', self.data_name,
                           self.model_name, folder_name)
        if not exists(plot_folder):
            makedirs(plot_folder)

        # get the data needed for plotting
        im = data_tpl['image']
        if 'fbp' in data_tpl:
            # in this case the image is not in HU, let's convert
            im = self.postprocessing(im)
        fbp = self.postprocessing(data_tpl['fbp'])
        pred = data_tpl[self.pred_key]
        # extract slices
        z = np.random.randint(im.shape[0])
        im_sl = im[z]
        pred_sl = pred[z]
        # compute fbp
        fbp_sl = fbp[z]
        # compute PSNR
        mse_fbp = np.mean((im_sl - fbp_sl) ** 2)
        mse_pred = np.mean((im_sl - pred_sl) ** 2)
        Imax2 = (im_sl - im_sl.min()).max()**2
        PSNR_pred = 10 * np.log10(Imax2 / mse_pred)
        PSNR_fbp = 10 * np.log10(Imax2 / mse_fbp)

        # now some plotting
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(fbp_sl, cmap='gray')
        plt.title('FBP: {:.2f}dB'.format(PSNR_fbp))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(im_sl, cmap='gray')
        plt.title('Target')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(pred_sl, cmap='gray')
        plt.title('CNN: {:.2f}dB'.format(PSNR_pred))
        plt.axis('off')
        plt.savefig(join(plot_folder, filename+'.png'), bbox_inches='tight')
        plt.close(fig)

    def compute_error_metrics(self, data_tpl):
        pred = data_tpl[self.pred_key].astype(float)
        im = data_tpl['image'].astype(float)
        if 'fbp' in data_tpl:
            # in this case the image is not in HU, let's convert
            im = self.postprocessing(im)
        mse = np.mean((pred - im)**2)
        Imax2 = (im - im.min()).max()**2
        PSNR = 10 * np.log10(Imax2 / mse) if mse > 0 else np.nan
        return {'PSNR': PSNR}

    def _init_global_metrics(self):
        self.global_metrics_helper = {'squared_error': 0,
                                      'Imax_squared': -np.inf, 'n': 0}
        self.global_metrics = {'PSNR': -1}

    def _update_global_metrics(self, data_tpl):
        im = self.postprocessing(data_tpl['image'])
        pred = data_tpl[self.pred_key]

        # get the helper variables
        sq_err = self.global_metrics_helper['squared_error']
        Imax_sq = self.global_metrics_helper['Imax_squared']
        n = self.global_metrics_helper['n']

        # update helper variables
        sq_err += np.sum((im - pred)**2) / 10**7
        n += np.prod(im.shape) / 10**7
        Imax_sq = max([Imax_sq, (im.max() - im.min())**2])

        # now update the metrics
        self.global_metrics['PSNR'] = 10 * np.log10(Imax_sq / sq_err)

        # and store the helpers again
        self.global_metrics_helper['squared_error'] = sq_err
        self.global_metrics_helper['Imax_squared'] = Imax_sq
        self.global_metrics_helper['n'] = n
