from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.label_utils import reduce_classes
import numpy as np
import torch
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from os.path import join, exists
from os import environ
from ovseg.utils.path_utils import maybe_create_path
from ovseg import OV_PREPROCESSED
from time import sleep
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x


class ContourRefinementPreprocessing(SegmentationPreprocessing):
    
    def __init__(self, *args, r1=None, r2=None, r_max=15, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert len(self.prev_stages) == 1, 'One previous stage must be given'
        self.r_max = r_max
        self.r1 = r1
        self.r2 = r2
        
        self.preprocessing_parameters.extend(['r1', 'r2'])
        
    def _get_selem(self, r):
        
        z_to_xy_ratio = self.target_spacing[0] / self.target_spacing[1]
        # radius in different directions
        rz = int(r/z_to_xy_ratio + 0.5)
        rxy = int(r)
        # set up ball
        selem = (np.sum(np.stack(np.meshgrid(*[np.linspace(-1, 1, 2*R+1) for R in [rz, rxy, rxy]], indexing='ij'))**2,0)<=1).astype(float)
        selem /= selem.sum()
        
        selem = torch.from_numpy(selem).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
        
        return selem
    
    def _eros(self, pred, r):
        
        
        if not torch.is_tensor(pred):
            pred = torch.from_numpy(pred)
            if torch.cuda.is_available():
                pred = pred.cuda()
        
        if len(pred.shape) < 5:
            while len(pred.shape) < 5:
                pred = pred.unsqueeze(0)
        
        
        z_to_xy_ratio = self.target_spacing[0] / self.target_spacing[1]
        # radius in different directions
        rz = int(r/z_to_xy_ratio + 0.5)
        rxy = int(r)

        pred_conv = torch.nn.functional.conv3d(pred,
                                               self.get_selem(r),
                                               padding=(rz,rxy,rxy))
    
        pred_eros = (pred_conv >= 1).type(torch.float)
        
        return pred_eros
    
    def _dial(self, pred, r):
        
        if not torch.is_tensor(pred):
            pred = torch.from_numpy(pred)
            if torch.cuda.is_available():
                pred = pred.cuda()
        
        if len(pred.shape) < 5:
            while len(pred.shape) < 5:
                pred = pred.unsqueeze(0)
        
        
        z_to_xy_ratio = self.target_spacing[0] / self.target_spacing[1]
        # radius in different directions
        rz = int(r/z_to_xy_ratio + 0.5)
        rxy = int(r)

        pred_conv = torch.nn.functional.conv3d(pred,
                                               self.get_selem(r),
                                               padding=(rz,rxy,rxy))
    
        pred_dial = (pred_conv > 0).type(torch.float)
        
        return pred_dial

    def _compute_r1_r2(self, data_tpl):
        
        r1,r2 = 0, 0
        
        # get the clean label (with reduced classes)
        lb = self.maybe_clean_label_from_data_tpl(data_tpl)
        assert lb.max() == 1 and lb.min() == 0, 'expected binary label'
    
        # get prediction, both should be binary
        pred = data_tpl[self.keys_for_previous_stages[0]]
        assert pred.max() == 1 and pred.min() == 0, 'expected binary prediction'
        
        # bring both to GPU
        lb_gpu = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).type(torch.float)
        pred_gpu = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).type(torch.float)
        
        # compute r1
        lb_vol = lb_gpu.sum().item()
        ovlp = (lb_gpu * pred_gpu).sum().item()
        sens = ovlp / lb_vol
        
        if lb_vol > 0:
            while sens < 0.995 and r1 < self.r_max:
                r1 += 1
                pred_dial = self._dial(pred_gpu, r1)
                ovlp = (lb_gpu * pred_dial).sum().item()
                sens = ovlp / lb_vol
        
        # compute r2
        pred_vol = pred_gpu.sum().item()
        
        ovlp = (lb_gpu * pred_gpu).sum().item()
        prec = ovlp / pred_vol
        
        if pred_vol > 0:
            while prec < 0.995:
                r2 += 1
                pred_eros = self._eros(pred, r2)
                pred_vol = pred_eros.sum().item()
                ovlp = (lb_gpu * pred_eros).sum().item()
                prec = ovlp / pred_vol
        
        return r1, r2

    def plan_preprocessing_raw_data(self,
                                    raw_data,
                                    percentiles=[0.5, 99.5],
                                    image_folder=None,
                                    dcm_revers=True,
                                    dcm_names_dict=None,
                                    force_planning=False,
                                    precentile_r=99):
        
        super().plan_preprocessing_raw_data(raw_data,
                                            percentiles=percentiles,
                                            image_folder=image_folder,
                                            dcm_revers=dcm_revers,
                                            dcm_names_dict=dcm_names_dict,
                                            force_planning=force_planning)
        
        if np.isscalar(self.r1) and np.isscalar(self.r2):
            return
        
        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')
        datasets = []
        for data_name in raw_data:
            print('Reading ' + data_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name),
                                 image_folder=image_folder,
                                 dcm_revers=dcm_revers,
                                 dcm_names_dict=dcm_names_dict,
                                 create_missing_labels_as_zero=True,
                                 prev_stages=self.prev_stages)
            datasets.append(raw_ds)
        
        # after all other parameters were infered, let's compute r1 and r2
        r_list = []
        
        print('Computing dilation and erosion parameters r1 and r2...')
        
        for dataset in datasets:
            
            for i in tqdm(range(len(dataset))):
                
                data_tpl = dataset[i]
                r_list.append(self._compute_r1_r2(data_tpl))
        
        # we add the 1 just for to be safe
        self.r1, self.r2 = np.percentile(r_list, precentile_r, 0) + 1
        
    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        # getting the image
        xb = data_tpl['image'].astype(float)

        # assuring the array is 4d
        xb = maybe_add_channel_dim(xb)

        # prediction from previous stage
        pred = data_tpl[self.keys_for_previous_stages[0]]
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()

        # ensure the array is 4d
        pred = maybe_add_channel_dim(pred)
        
        xb = np.concatenate([xb, pred])

        if 'label' in data_tpl and not get_only_im:     
            # get the label from the data_tpl and clean if applicable
            lb = self.maybe_clean_label_from_data_tpl(data_tpl)

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis].astype(float)
            xb = np.concatenate([xb, lb])
        
        # finally add batch axis
        xb = xb[np.newaxis]

        return xb
    
    def __call__(self, data_tpl, preprocess_only_im=False, return_np=False):
        
        xb_prep = super().__call__(data_tpl, preprocess_only_im, return_np)
        
        pred = xb_prep[self.n_im_channels:self.n_im_channels+1]
        
        # create the mask by dilation and erosing (after preprocessing)
        pred_dial = self._dial(pred, self.r1)[0]
        pred_eros = self._eros(pred, self.r2)[0]
        mask = pred_dial - pred_eros
        
        if return_np:
            mask = mask.cpu().numpy()
            xb_prep = np.concatenate([xb_prep[:self.n_im_channels],
                                      mask,
                                      xb_prep[-1:]])
        else:
            xb_prep = torch.cat([xb_prep[:self.n_im_channels],
                                 mask,
                                 xb_prep[-1:]])
        
        return xb_prep

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
                                 prev_stages=self.prev_stages)
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
        for f in ['images', 'labels', 'fingerprints', 'masks', 'prev_preds']:
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
                prev_pred = xb[self.n_im_channels:self.n_im_channels+1].astype(np.uint8)
                mask = xb[-2:-1].astype(np.uint8)
                lb = xb[-1:].astype(np.uint8)

                if lb.max() == 0 and self.save_only_fg_scans:
                    continue
                
                if mask.max() == 0:
                    # if the whole mask is zero there is nothing to do
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
                np.save(join(outfolder, 'prev_preds', scan), prev_pred)
                np.save(join(outfolder, 'masks', scan), mask)
                np.save(join(outfolder, 'fingerprints', scan), fingerprint)