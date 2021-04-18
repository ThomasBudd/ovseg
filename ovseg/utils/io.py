import numpy as np
import nibabel as nib
import pydicom
from os.path import join, exists, basename, split
from os import listdir, environ
try:
    from skimage.draw import polygon
except ImportError:
    print('Caught Import Error while importing some function from scipy or skimage. '
          'Please use a newer version of gcc.')
import pickle

_names_sorting_warning_printed = False
_names_dict_warning_printed = False
_isotropic_volume_loaded_warning_printed = False
_ananisotropic_volume_loaded_warning_printed = False


def load_pkl(path_to_file):
    with open(path_to_file, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pkl(data, path_to_file):
    if not path_to_file.endswith('.pkl'):
        path_to_file += '.pkl'
    with open(path_to_file, 'wb') as file:
        pickle.dump(data, file)


def save_txt(data, path_to_file):
    if not path_to_file.endswith('.txt'):
        path_to_file += '.txt'

    dict_name = basename(path_to_file)[:-4]
    with open(path_to_file, 'w') as file:
        _write_dict_to_txt(dict_name, data, file, n_tabs=0)


def _write_dict_to_txt(self, dict_name, data, file, n_tabs):
    # recurively go down all dicts and print their content
    # each time we go a dict deeper we add another tab for more beautiful
    # nested printing
    tabs = ''.join(n_tabs * ['\t'])
    s = tabs + dict_name + ' =\n'
    file.write(s)
    for key in data.keys():
        item = data[key]
        if isinstance(item, dict):
            self._write_dict_to_txt(key, item, file, n_tabs+1)
        else:
            s = tabs + '\t' + key + ' = ' + str(item) + '\n'
            file.write(s)


def read_nii(nii_file):
    global _isotropic_volume_loaded_warning_printed, _ananisotropic_volume_loaded_warning_printed
    img = nib.load(nii_file)
    spacing = img.header['pixdim'][1:4]
    im = img.get_fdata()
    dims = img.shape
    # first check if the z axis is last or first
    if spacing[0] == spacing[1]:
        if spacing[0] != spacing[2]:
            has_z_first = False
        elif dims[0] == dims[1] and dims[0] != dims[2]:
            # in the case of isotropic voxel we check the dimensions of the image instead
            has_z_first = False
        elif dims[1] == dims[2] and dims[0] != dims[2]:
            has_z_first = True
        else:
            if not _isotropic_volume_loaded_warning_printed:
                print('Found at least one file {} with isotropic voxel and equal volume dimensions.'
                      'could not infere if the z axis is first or last, guessing last. '
                      'Please make sure it is!'.format(nii_file))
                has_z_first = False
                _isotropic_volume_loaded_warning_printed = True
    else:
        # spacing[0] != spacing[1]
        if spacing[1] == spacing[2]:
            has_z_first = True
        else:
            if not _ananisotropic_volume_loaded_warning_printed:
                print('Found at least one file {} with voxelspacing {}. '
                      'Need at least two equal numbers in the spacing to find out if the z '
                      'axis is first or last, guessing last. Please make sure it is!'
                      ''.format(nii_file, spacing))
                has_z_first = False
                _ananisotropic_volume_loaded_warning_printed = True

    # now we (hopefully) know if the z axis is first or last
    if not has_z_first:
        # z axis is in the back, get it to the front!
        spacing = np.array([spacing[2], spacing[0], spacing[1]])
        im = np.moveaxis(im, 2, 0)

    return im, spacing, has_z_first


def read_nii_files(nii_files):
    '''
    read_nii_files(*args)

    reads nii files that belong togehter, e.g. image channels and a
    corresponding segmentation.

    Parameters
    ----------
    *args : list
        full pathes to nii files to be read

    Raises
    ------
    ValueError
        if images found do not have the same voxel spacing

    Returns
    -------
    image
        4d tensor with images stacked up in first (channel) dimension
    spacing : len 3
        voxel spacing in x, y, z direction in mm

    '''
    im, spacing, had_z_first = read_nii(nii_files[0])
    out_volumes = [im]
    for nii_file in nii_files[1:]:
        out = read_nii(nii_file)
        if np.any(out[1] != spacing):
            raise ValueError('Spacing didn\'t match for '
                             + nii_files[0] + ' and ' + nii_file + '. Got '
                             + str(spacing) + ' and ' + str(out[1]))
        out_volumes.append(out[0])
    return np.stack(out_volumes), spacing


def read_data_tpl_from_nii(folder, case):
    data_tpl = {}

    if not exists(folder):
        folder = join(environ['OV_DATA_BASE'], 'raw_data', folder)

    if not exists(folder):
        raise FileNotFoundError('Can\'t read from folder {}. It doesn\'t exist.'.format(folder))

    if isinstance(case, int):
        case = 'case_%03d.nii.gz' % case

    if not isinstance(case, str):
        raise TypeError('Input \'case\' must be string, not {}'.format(type(case)))

    if not case.endswith('.nii.gz'):
        case += '.nii.gz'

    # first let's read the data info
    if exists(join(folder, 'data_info.pkl')):
        data_info = load_pkl(join(folder, 'data_info.pkl'))
        if case[:-7] in data_info:
            # the [:-7] is to remove the .nii.gz
            data_tpl.update(data_info[case[:-7]])

    # first check if the image folder exists
    possible_image_folders = ['images', 'imagesTr', 'imagesTs']
    image_folders_ex = [join(folder, imf) for imf in possible_image_folders
                        if exists(join(folder, imf))]
    if len(image_folders_ex) == 0:
        raise FileNotFoundError('Didn\'t find any image folder in {}.'.format(folder))

    image_files = []
    for image_folder in image_folders_ex:
        matching_files = [join(folder, image_folder, file) for file in
                          listdir(join(folder, image_folder))
                          if file.startswith(case[:-7])]
        if len(matching_files) > 0 and len(image_files) > 0:
            raise FileExistsError('Found images for in multiple image folders at path {} for '
                                  'case {}.'.format(folder, case))
        image_files = matching_files

    if len(image_files) == 0:
        raise FileNotFoundError('No image files found for case {}.'.format(case))
    elif len(image_files) == 1:
        raw_image_file = image_files[0]
        im, spacing, had_z_first = read_nii(raw_image_file)
    else:
        raw_image_file = image_files
        im_data = [read_nii(file) for file in raw_image_file]
        ims = [im for im, spacing, had_z_first in im_data]
        spacings = [spacing for im, spacing, had_z_first in im_data]
        hzf_list = [had_z_first for im, spacing, had_z_first in im_data]

        # now check if everything matches
        if not np.all([np.all(spacings[0] == sp) for sp in spacings[1:]]):
            raise ValueError('Found unequal spacings when reading the image files {}'
                             ''.format(image_files))

        if not np.all([np.all(hzf_list[0] == hzf) for hzf in hzf_list[1:]]):
            raise ValueError('Found some files with the z axis first and some with z axis last '
                             'when reading the image files {}'.format(image_files))

        im = np.stack(ims)
        spacing = spacings[0]
        had_z_first = hzf_list[0]
    data_tpl['image'] = im
    data_tpl['spacing'] = spacing
    data_tpl['had_z_first'] = had_z_first
    data_tpl['raw_image_file'] = raw_image_file

    label_folders_ex = [join(folder, lbf) for lbf in ['labels', 'labelsTr', 'labelsTs']
                        if exists(join(folder, lbf))]

    if len(label_folders_ex) == 0:
        # when there are no existing label folders we can just return the data tpl
        return data_tpl

    label_files = []
    for label_folder in label_folders_ex:
        matching_files = [join(folder, label_folder, file) for file in
                          listdir(join(folder, label_folder))
                          if file.startswith(case[:-7])]
        if len(matching_files) > 0 and len(label_files) > 0:
            raise FileExistsError('Found labels for in multiple label folders at path {} for '
                                  'case {}.'.format(folder, case))
        label_files = matching_files
    if len(label_files) == 0:
        # in case we don't find a label file let's return without
        return data_tpl
    elif len(label_files) == 1:
        lb, spacing, had_z_first = read_nii(label_files[0])
        if not np.all(spacing == data_tpl['spacing']):
            raise ValueError('Found not matching spacings for case {}.'.format(case))
        if had_z_first != data_tpl['had_z_first']:
            raise ValueError('Axis ordering doesn\'t match for case {}'
                             'Make sure image and label files have the z axis at the same position'
                             '(first or last).'.format(case))
        data_tpl['label'] = lb
        data_tpl['raw_label_file'] = label_files[0]
    else:
        raise FileExistsError('Found multiple label files for case {}'.format(case))

    return data_tpl


def _is_im_dcm_ds(ds):
    attrs = ['pixel_array', 'ImagePositionPatient', 'PixelSpacing',
             'RescaleSlope', 'RescaleIntercept']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def _is_roi_dcm_ds(ds):
    attrs = ['StructureSetROISequence', 'ROIContourSequence']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def read_dcms(dcm_folder, reverse=True, names_dict=None, dataset=None):
    '''
    read_dcms(dcms, dcmrt=None, reverse=True, names_dict=None)

    Reads dicom files for axial images and dcmrt files that contain ROIS.
    If is assumed that both the image dcms and the dcmrt dicoms are files in
    \'dcm_folder\'. If not dcmrt files is found an empty segmentation is
    returned, else one segmentation array is created for each dcmrt file.
    Image dcms should be of axial reconstructions with the attributes
    pixel_array, ImagePositionPatient, PixelSpacing, RescaleSlope,
    RescaleIntercept, roi dcms should have the attributes
    StructureSetROISequence and ROIContourSequence

    Parameters
    ----------
    dcm_folder : str
        full path of dcm_folder to dcms to be read
    reverse : bool, optional
        For right handed coordinate systems like Siemens uses them the top
        slice has the highest z value, so in this case we sort the dcms in
        descending (reverse) order with respect to their z coordinate.
        The default is True.
    names : list
        names that are contained in the ROI file. Will be used to encode the
        ROIs in the output seg, e.g. the ROI names[0] will be encoded as 1
        in the output and so on. If the list is not given it is checked if
        all names start with a number, otherwise all occuring names is sorted
        alphabetically. Comparisson of names if case insensitive

    Returns
    -------
    data_tpl

    '''
    global _names_sorting_warning_printed, _names_dict_warning_printed
    # read the image and sort it with respect to the z coordinates
    dcms = [join(dcm_folder, dcm) for dcm in listdir(dcm_folder)]
    dcms.sort()
    imdss = []
    roidss = []
    roidcms = []
    for dcm in dcms:
        ds = pydicom.dcmread(dcm)
        if _is_im_dcm_ds(ds):
            imdss.append(ds)
        elif _is_roi_dcm_ds(ds):
            roidss.append(ds)
            roidcms.append(dcm)
        else:
            raise TypeError(dcm + ' is neither image nor roi dcm.')
    if len(roidss) > 1:
        raise FileExistsError('Found multiple ROI dcms in folder '+dcm_folder+'. '
                              'Make sure that at most one ROI dcm file is in each folder.')
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]
    imdss = [ds for _, ds in sorted(zip(z_im, imdss), reverse=reverse)]
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]

    # now get the spacing
    ps = np.array(imdss[0].PixelSpacing).astype(float)
    z_sp = np.abs(np.median(np.diff(z_im)))
    spacing = np.array([z_sp, *ps])

    # convert the image in HU
    a = imdss[0].RescaleSlope
    b = imdss[0].RescaleIntercept
    # stack up the image at the first axes
    im = np.stack([imds.pixel_array*a+b for imds in imdss], 0).astype(np.int16)
    im[im < -1024] = -1024

    if len(roidss) == 1:
        roids = roidss[0]
        roidcm = roidcms[0]
        seg = np.zeros_like(im, dtype=np.uint8)

        pos_r = float(imdss[0].ImagePositionPatient[1])
        pos_c = float(imdss[0].ImagePositionPatient[0])
        names_found = [s.ROIName.lower() for s in roids.StructureSetROISequence]
        names_found = np.unique(names_found).tolist()
        names_found.sort()
        if names_dict is None:
            if np.all([name[0].isdigit() for name in names_found]):
                # lucky case! All our ROIs start with numbers
                names_dict = {}
                for name in names_found:
                    i = 0
                    while name[:i+1].isdigit():
                        i += 1
                        if i == len(name):
                            break
                    num = int(name[:i])
                    names_dict[name] = num
            else:
                # this is not so good.... if the ROIs don't start with numbers we have to make an
                # elaborate guess on which index which name belongs to
                if not _names_dict_warning_printed:
                    print('Warning: No names_dict was found and the ROIs do not start with integer '
                          'numbers. The mapping of which integer in the labels belong to which '
                          'ROI is now based on sorting the ROI names. If not all scans have the '
                          'same ROIs with the same names this might lead to errors in the '
                          'labeling.')
                    _names_dict_warning_printed = True
                names_dict = {name: i for i, name in enumerate(names_found)}
        else:
            for name in names_found:
                if name not in names_dict:
                    raise ValueError('Name error in '+roidcm+'. Found ROI with'
                                     ' name '+name+' which was not given in'
                                     ' the names_dict.')
        # now let's look at all ROIS
        for i in range(len(roids.ROIContourSequence)):
            name = roids.StructureSetROISequence[i].ROIName.lower()
            num = names_dict[name]
            for s in roids.ROIContourSequence[i].ContourSequence:
                c = s.ContourData
                # list of polygone corners
                nodes = np.array(c).reshape((-1, 3))
                ad = np.abs(z_im-nodes[0, 2])
                # z index of the slice the contour is marked in
                z_index = np.argmin(np.abs(ad))
                if np.max(ad) > 0.1:
                    ValueError('z axis of difference larger than .1mm found'
                               ' between dcmrt and dcms.')
                r = (nodes[:, 1] - pos_r) / spacing[2]
                # from patient coordinate system to index of the image
                c = (nodes[:, 0] - pos_c) / spacing[1]
                rr, cc = polygon(r, c)
                seg[z_index, rr, cc] = num

    data_tpl = {}
    data_tpl['image'] = im
    data_tpl['raw_image_file'] = dcm_folder
    if len(roidcms) > 0:
        data_tpl['label'] = seg
        if len(roidcms) == 1:
            roidcms = roidcms[0]
        data_tpl['raw_label_file'] = roidcms
    data_tpl['spacing'] = spacing
    data_tpl['z_pos'] = z_im
    try:
        data_tpl['SOP_ids'] = [ds.SOPInstanceUID for ds in imdss]
    except AttributeError:
        print('Warning: at least on SOPInstanceUID is missing for the dcm files in {}. '
              'This means that the results can not be saved as dcm rt files, but will be saved '
              'as nifti.'.format(dcm_folder))
    ds = imdss[0]
    for key, attr in zip(['pat_id', 'date', 'pat_name'],
                         ['PatientID', 'AcquisitionDate', 'PatientName']):
        if hasattr(ds, attr):
            data_tpl[key] = str(ds.__getattr__(attr))

    data_tpl['dataset'] = basename(split(dcm_folder)[0]) if dataset is None else dataset

    return data_tpl


def save_nii(im, out_file, spacing=None, img=None):
    '''
    save_nii(im, out_file, spacing=None, img=None)

    saves image as nii file by either overwriting image data from another nii
    file or by giving the spacing and creating a new nii file

    Parameters
    ----------
    im : 3d array
        image data.
    out_file : str
        path to where the image should be stored
    spacing : len 3
        voxel spacing in mm, optional
    img : nifti image
        Image to be used for overwriting.

    Returns
    -------
    None.

    '''
    if not out_file.endswith('.nii.gz'):
        out_file = out_file + '.nii.gz'

    if spacing is None and img is None:
        raise ValueError('Voxel spacing or another nifti image must be given'
                         'as input when writing a new nifti file.')
    elif spacing is None:
        im_nii = nib.Nifti1Image(im, img.affine, img.header)
    else:
        im_nii = nib.Nifti1Image(im, np.eye(4))
        im_nii.header['pixdim'][1:4] = spacing
    nib.save(im_nii, out_file)
