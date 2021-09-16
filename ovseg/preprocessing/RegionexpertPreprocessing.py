from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.label_utils import reduce_classes
import numpy as np
import torch
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from os.path import join, exists
from os import environ
from ovseg.utils.path_utils import maybe_create_path
from time import sleep
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x


class RegionexpertPreprocessing(SegmentationPreprocessing):

    def __init__(self, *args, region_finding_model:dict, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.region_finding_model = region_finding_model
        
        for key in ['data_name', 'preprocessed_name', 'model_name']:
            assert key in self.region_finding_model
        self.region_finding_key = '_'.join(['prediction',
                                            self.region_finding_model['data_name'],
                                            self.region_finding_model['preprocessed_name'],
                                            self.region_finding_model['model_name']])

        self.prev_stages = [self.region_finding_model]

        self.preprocessing_parameters = ['apply_resizing',
                                         'apply_pooling',
                                         'apply_windowing',
                                         'target_spacing',
                                         'pooling_stride',
                                         'window',
                                         'scaling',
                                         'lb_classes',
                                         'reduce_lb_to_single_class',
                                         'lb_min_vol',
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'prev_stages',
                                         'dataset_properties',
                                         'region_finding_model']

    
    
    def maybe_clean_region_from_data_tpl(self, data_tpl):

        if self.region_finding_key not in data_tpl:
            raise ValueError('Can\'t clean region from data tpl, none was found!')

        reg = data_tpl[self.region_finding_key]

        if self.is_preprocessed_data_tpl(data_tpl):
            return reg

        if self.lb_classes is not None:
            # get only the relevant classes from the region (and reduce to a binary label)
            reg = reduce_classes(reg, self.lb_classes, False)

        return reg
    
    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        # getting the image
        xb = data_tpl['image'].astype(float)

        # assuring the array is 4d
        xb = maybe_add_channel_dim(xb)

        key = self.region_finding_key
        assert key in data_tpl, 'prediction '+key+' from previous stage missing'
        
        # all regions from the data_tpl
        reg = self.maybe_clean_region_from_data_tpl(data_tpl)
        if len(reg.shape) == 3:
            reg = reg[np.newaxis]
        reg = reg.astype(float)
        xb = np.concatenate([xb, reg])

        if 'label' in data_tpl and not get_only_im:     
            # get the label from the data_tpl and clean if applicable
            lb = self.maybe_clean_label_from_data_tpl(data_tpl)

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis].astype(float)
            xb = np.concatenate([xb, lb])
        
        # finally add batch axis
        xb = xb[np.newaxis]

        return xb

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
                                 prev_stages=[self.region_finding_model])
            raw_ds_list.append(raw_ds)

        if not self.is_initalised:
            print('Preprocessing classes were not initialised when calling '
                  '\'preprocess_raw_data\'. Doing it now.\n')
            self.initialise_preprocessing()

        im_dtype = np.float16 if save_as_fp16 else np.float32

        if data_name is None:
            data_name = '_'.join(sorted(raw_data))

        # root folder of all saved preprocessed data
        outfolder = join(environ['OV_DATA_BASE'], 'preprocessed', data_name, preprocessed_name)
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', data_name, preprocessed_name)
        print(outfolder, plot_folder)
        # now let's create the output folders
        # reg_folder = self.region_finding_key[:10] + 's' + self.region_finding_key[10:]
        reg_folder = 'regions'
        for f in ['images', 'labels', 'fingerprints', reg_folder]:
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
                try:
                    # sometimes we will have no region file for the scan, in this case
                    # we're skipping (doesn't make a difference if save_only_fg_scans is true)
                    data_tpl = raw_ds[i]
                except FileNotFoundError:
                    continue

                im, spacing = data_tpl['image'], data_tpl['spacing']

                orig_shape = im.shape[-3:]
                orig_spacing = spacing.copy()

                # some data_tpls come without labels e.g. from dcms if there are no ROIs
                if 'label' not in data_tpl:
                    data_tpl['label'] = np.zeros(orig_shape)

                # get the preprocessed volumes from the data_tpl
                xb = self.__call__(data_tpl, return_np=True)
                im = xb[:self.n_im_channels].astype(im_dtype)
                reg = xb[self.n_im_channels:self.n_im_channels+1].astype(np.uint8)
                lb = xb[self.n_im_channels+1:self.n_im_channels+2].astype(np.uint8)

                if lb.max() == 0 and self.save_only_fg_scans:
                    continue
                
                # if reg.max() == 0:
                #     # we don't need to save images without foreground, there is nothing to do
                #     # here
                #     continue

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
                np.save(join(outfolder, reg_folder, scan), reg)
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