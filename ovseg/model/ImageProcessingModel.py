import numpy as np
import torch
from os.path import join, basename
from ovseg.utils.io import save_nii
import matplotlib.pyplot as plt
from ovseg.training.ImageProcessingTraining import \
    ImageProcessingTraining
from ovseg.data.ImageProcessingData import ImageProcessingData
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type
from ovseg.preprocessing.ImageProcessingPreprocessing import \
    ImageProcessingPreprocessing
from ovseg.networks.UNet import UNet


class ImageProcessingModel(ModelBase):

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', batch_size_val=4,
                 fp32_val=False, plot_window=[-50, 350]):
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
        preprocessing_kwargs = self.model_parameters['preprocessing']
        self.preprocessing = ImageProcessingPreprocessing(**preprocessing_kwargs)
        print('preprocessing initialised')

    def initialise_augmentation(self):
        print('no augmentation needed')

    def initialise_network(self):
        if 'network' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'network\'. These must contain the '
                                 'dict of network paramters.')
        params = self.model_parameters['network'].copy()
        self.network = UNet(**params).cuda()
        print('Network initialised')

    def initialise_postprocessing(self):
        self.window = self.preprocessing.window
        print('postprocessing initialised')

    def initialise_data(self):
        print('initialise data')
        try:
            params = self.model_parameters['data']
        except KeyError:
            params = {}
            print('Warning! No data parameters found')

        self.data = ImageProcessingData(self.val_fold, self.preprocessed_path,
                                        **params)

    def initialise_training(self):
        print('Initialise training')
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        params = self.model_parameters['training'].copy()

        self.training = ImageProcessingTraining(network=self.network,
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
        im_HU = im * (self.window[1] - self.window[0]) + self.window[0]

        return im_HU

    def predict(self, data_dict, return_torch=False):
        '''
        predict(proj)

        evaluates the projection data of a full scan and return the 3d image

        '''
        im = data_dict['image']
        is_np, _ = check_type(im)

        # make sure im is torch tensor
        if is_np:
            im = torch.from_numpy(im).cuda()
        else:
            im = im.cuda()

        # add channel axes
        im = im.unsqueeze(0)

        nz = im.shape[-1]
        bs = self.batch_size_val
        pred = torch.zeros((512, 512, nz), device='cuda')
        z_list = list(range(0, nz - bs, bs)) + [nz - bs]

        # do the iterations
        with torch.no_grad():
            for z in z_list:
                batch = torch.stack([im[..., zb] for zb in range(z, z + bs)])
                if self.fp32_val:
                    out = self.network(batch)
                else:
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)[0]
                # out back to gpu and reshape
                out = torch.stack([out[b, 0] for b in range(bs)], -1)
                pred[..., z: z + bs] = out

            # convert to HU
            pred = self.postprocessing(pred)

        if is_np and not return_torch:
            pred = pred.cpu().numpy()

        torch.cuda.empty_cache()

        return pred

    def save_prediction(self, pred, data, pred_folder, name):
        save_nii(pred, join(pred_folder, basename(data['raw_label_file'])), data['spacing'])

    def plot_prediction(self, pred, data, plot_folder, name):
        im = self.postprocessing(data['image']).clip(*self.plot_window)
        # extract slices
        z = np.random.randint(im.shape[-1])
        im_sl = im[..., z]
        pred_sl = pred[..., z].clip(*self.plot_window)
        # compute PSNR
        mse_pred = np.mean((im_sl - pred_sl) ** 2)
        Imax2 = (im_sl - im_sl.min()).max()**2
        PSNR_pred = 10 * np.log10(Imax2 / mse_pred)

        # now some plotting
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(im_sl, cmap='gray',
                   vmin=self.plot_window[0],
                   vmax=self.plot_window[1])
        plt.title('Target')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(pred_sl, cmap='gray',
                   vmin=self.plot_window[0],
                   vmax=self.plot_window[1])
        plt.title('CNN: {:.2f}dB'.format(PSNR_pred))
        plt.axis('off')
        plt.savefig(join(plot_folder, name+'.png'), bbox_inches='tight')
        plt.close(fig)

    def compute_error_metrics(self, pred, data):
        im = self.postprocessing(data['image'])
        mse = np.mean((pred - im)**2)
        Imax2 = (im - im.min()).max()**2
        PSNR = 10 * np.log10(Imax2 / mse)
        rel_mse = mse / np.mean(im**2)
        im_win = im.clip(*self.plot_window)
        pred_win = pred.clip(*self.plot_window)
        mse_win = np.mean((pred_win - im_win)**2)
        Imax2_win = (im_win - im_win.min()).max()**2
        PSNR_win = 10 * np.log10(Imax2_win / mse_win)
        return {'PSNR': PSNR, 'PSNR_win': PSNR_win, 'rel_mse': rel_mse}
