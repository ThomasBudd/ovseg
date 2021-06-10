from ovseg.preprocessing.Restauration2dSimPreprocessing import Restauration2dSimPreprocessing
from os.path import join
from os import environ
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

# first some arguments we need for the planning and saving
parser.add_argument("raw_data", nargs='+',
                    help='Name or names of folders in \'raw_data\' that are used for planning '
                    'and that are preprocessed.')
parser.add_argument("preprocessed_name",
                    help='Name that the folder in \'preprocessed\' will have.')
parser.add_argument("ext", help='extension of the folder names e.g. \'half\'.')
parser.add_argument("--save_as_fp32", required=False, default=False, action='store_true',
                    help='Usually the preprocessed images are stored as fp16 to save disk space. '
                    'Set this if you are sure that you need them to be stored as fp32 instead.')
parser.add_argument("--dont_apply_resizing", required=False, default=False, action='store_true')
parser.add_argument("--dont_apply_windowing", required=False, default=False, action='store_true')
parser.add_argument("--target_z_spacing", required=False, default=5.0, type=float)
parser.add_argument("--n_angles", required=False, default=500, type=float)
parser.add_argument("--source_distance", required=False, default=600, type=float)
parser.add_argument("--det_count", required=False, default=736, type=float)
parser.add_argument("--det_spacing", required=False, default=1.0, type=float)
parser.add_argument("--dose_level", required=False, default=1.0, type=float)
parser.add_argument("--window", required=False, default=None, nargs='+')
parser.add_argument("--scaling", required=False, default=None, nargs='+')


args = parser.parse_args()

if args.window is None:
    window = [-32, 318]
else:
    window = [float(w) for w in args.window]
    assert len(window) == 2

if args.dont_apply_windowing:
    window = None

if args.scaling is None:
    scaling = [52.286, 38.16]
else:
    scaling = [float(s) for s in args.scaling]
    assert len(scaling) == 2


preprocesseing = Restauration2dSimPreprocessing(n_angles=args.n_angles,
                                                source_distance=args.source_distance,
                                                det_count=args.det_count,
                                                det_spacing=args.det_spacing,
                                                mu_water=0.0192,
                                                window=window,
                                                scaling=scaling, fbp_filter='ramp',
                                                apply_z_resizing=True,
                                                target_z_spacing=args.target_z_spacing,
                                                bowtie_filt=None, dose_level=args.dose_level)

if len(args.ext) > 0:
    fbp_folder_name = 'fbps_'+args.ext
    im_folder_name = 'images_restauration_'+args.ext
else:
    fbp_folder_name = 'fbps'
    im_folder_name = 'images_restauration'



preprocesseing.preprocess_raw_folders(args.raw_data, args.preprocessed_name,
                                      fbp_folder_name=fbp_folder_name,
                                      im_folder_name=im_folder_name,
                                      save_as_fp16=not args.save_as_fp32)