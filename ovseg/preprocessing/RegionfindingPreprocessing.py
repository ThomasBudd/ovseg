from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.seg_fg_dial import seg_fg_dial
import numpy as np
import torch
from ovseg.utils.path_utils import maybe_create_path
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.data.Dataset import raw_Dataset
from ovseg import OV_PREPROCESSED
from os.path import join
from os import environ
import matplotlib.pyplot as plt
from time import sleep
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x

class RegionfindingPreprocessing(SegmentationPreprocessing):

    def __init__(self, r, z_to_xy_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.r = r
        self.z_to_xy_ratio = z_to_xy_ratio
    
        self.preprocessing_parameters.append('r')
        self.preprocessing_parameters.append('z_to_xy_ratio')

    def seg_to_region(self, seg):
        return seg_fg_dial(seg, r=self.r, z_to_xy_ratio=self.z_to_xy_ratio)
    
    def __call__(self, data_tpl, preprocess_only_im=False, return_np=False):
        volume = super().__call__(data_tpl, preprocess_only_im, return_np=True)

        if preprocess_only_im:
            if not return_np:
                volume = torch.tensor(volume)
                if torch.cuda.is_available():
                    volume = volume.cuda()
            return volume

        # now let's add the region to the preprocessed volumes
        region = self.seg_to_region(volume[-1])
        volume = np.concatenate([volume, region[np.newaxis]])

        if not return_np:
            volume = torch.tensor(volume)
            if torch.cuda.is_available():
                volume = volume.cuda()

        return volume
            
    def preprocess_raw_data(self,
                            raw_data,
                            preprocessed_name,
                            data_name=None,
                            save_as_fp16=True,
                            image_folder=None,
                            dcm_revers=True,
                            dcm_names_dict=None):

        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')

        print()
        raw_ds_list = []
        for raw_name in raw_data:
            print('Converting ' + raw_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', raw_name),
                                 image_folder=image_folder,
                                 dcm_revers=dcm_revers,
                                 dcm_names_dict=dcm_names_dict,
                                 prev_stages=self.prev_stages if self.is_cascade() else None)
            raw_ds_list.append(raw_ds)

        if not self.is_initalised:
            print('Preprocessing classes were not initialised when calling '
                  '\'preprocess_raw_data\'. Doing it now.\n')
            self.initialise_preprocessing()

        im_dtype = np.float16 if save_as_fp16 else np.float32

        if data_name is None:
            data_name = '_'.join(sorted(raw_data))

        # root folder of all saved preprocessed data
        outfolder = join(OV_PREPROCESSED, data_name, preprocessed_name)
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', data_name, preprocessed_name)
        print(outfolder, plot_folder)
        # now let's create the output folders
        for f in ['images', 'labels', 'fingerprints', 'regions']:
            maybe_create_path(join(outfolder, f))
        maybe_create_path(plot_folder)

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.maybe_save_preprocessing_parameters(outfolder)
        # here is the fun
        for raw_ds in raw_ds_list:
            print()
            sleep(1)
            for i in tqdm(range(len(raw_ds))):
                # read files
                data_tpl = raw_ds[i]

                im, spacing = data_tpl['image'], data_tpl['spacing']

                orig_shape = im.shape[-3:]
                orig_spacing = spacing.copy()

                # some data_tpls come without labels e.g. from dcms if there are no ROIs
                if 'label' not in data_tpl:
                    data_tpl['label'] = np.zeros(orig_shape)

                # get the preprocessed volumes from the data_tpl
                xb = self.__call__(data_tpl, return_np=True)
                im = xb[:self.n_im_channels].astype(im_dtype)
                lb = xb[self.n_im_channels:self.n_im_channels+1].astype(np.uint8)
                reg = xb[self.n_im_channels+1:self.n_im_channels+2].astype(np.uint8)

                if lb.max() == 0 and self.save_only_fg_scans:
                    continue

                spacing = self.target_spacing if self.apply_resizing else spacing
                if self.apply_pooling:
                    spacing = np.array(spacing) * np.array(self.pooling_stride)
                # the fingerprints are defined as everything that is left in the data_tpl
                # that is not image, label or prediction from a previous stage
                fingerprint_keys = [key for key in data_tpl if key not in ['image', 'label']]
                fingerprint_keys = [key for key in fingerprint_keys
                                    if not key.startswith('prediction')]
                fingerprint = {key: data_tpl[key] for key in fingerprint_keys}
                fingerprint['orig_shape'] = orig_shape
                fingerprint['orig_spacing'] = orig_spacing
                fingerprint['spacing'] = spacing
                scan = data_tpl['scan']
                if 'dataset' not in fingerprint:
                    fingerprint['dataset'] = raw_name
                if 'pat_id' not in fingerprint:
                    fingerprint['pat_id'] = scan
                # save image and label
                # remeber that the label carries all labels incl. potential masks or
                # predictions from previous stages
                np.save(join(outfolder, 'images', scan), im)
                np.save(join(outfolder, 'labels', scan), lb)
                np.save(join(outfolder, 'regions', scan), reg)
                np.save(join(outfolder, 'fingerprints', scan), fingerprint)

                # additionally do some plots
                im = im.astype(float)
                im_ch = im.shape[0]

                # get z values of interesting slices
                contains = np.where(np.sum(lb[0], (1, 2)))[0]
                z_list = [np.argmax(np.sum(lb[0], (1, 2)))]
                s_list = ['_largest', '_random']
                                
                if len(contains) > 0:
                    z_list.extend(np.random.choice(contains, size=1))
                else:
                    z_list.extend(np.random.randint(lb.shape, size=1))
                for z, s in zip(z_list, s_list):
                    fig = plt.figure()
                    for ic in range(im_ch):
                        plt.subplot(1, im_ch, ic+1)
                        plt.imshow(im[ic, z], cmap='gray')
                        if lb[0, z].max() > 0:
                            # this if is purely to avoid annoying UserWarning messages that
                            # interrupt the beautiful beautiful tqdm bar
                            plt.contour(lb[0, z] > 0,
                                        linewidths=0.5,
                                        colors='red',
                                        linestyles='solid')
                            
                        if reg[0, z].max() > 0:
                            plt.contour(reg[0, z] > 0,
                                        linewidths=0.25,
                                        colors='red',
                                        linestyles='dashed')

                        plt.axis('off')
                    plt.savefig(join(plot_folder, scan + s + '.png'))
                    plt.close(fig)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Preprocessing done!')