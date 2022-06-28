from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
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
parser.add_argument("--data_name", required=False, default=None,
                    help='set this in case you want to give the parent folder of the preprocessed '
                    'data a special name. If unset this name will be the name given in raw_data.')
parser.add_argument("--save_as_fp32", required=False, default=False, action='store_true',
                    help='Usually the preprocessed images are stored as fp16 to save disk space. '
                    'Set this if you are sure that you need them to be stored as fp32 instead.')
parser.add_argument("--save_scans_without_fg", required=False, default=False, action='store_true',
                    help='By default images that contain foreground are preprocessed and '
                    'used for training. Set this flag to undo so.')

# now the arguments that determine the preprocessing
parser.add_argument("--dont_apply_resizing", required=False, default=False, action='store_true')
parser.add_argument("--dont_apply_windowing", required=False, default=False, action='store_true')
parser.add_argument("--target_spacing", required=False, default=None, nargs='+')
parser.add_argument("--pooling_stride", required=False, default=None, nargs='+')
parser.add_argument("--window", required=False, default=None, nargs='+')
parser.add_argument("--scaling", required=False, default=None, nargs='+')
parser.add_argument("--lb_classes", required=False, default=None, nargs='+',
                    help='If you have a multiclass problem with a lot of classes and each model '
                    'is only supposed to use a subset of these, determine here which classes '
                    'you want to consider for this batch of preprocessed data.')
parser.add_argument("--reduce_lb_to_single_class", required=False, default=False,
                    action='store_true',
                    help='If you are in a multiclass setting and you wishes to reduce your '
                    'problem to a simply binary foreground vs. background setting, set this flag.')
parser.add_argument("--lb_min_vol", required=False, default=None, nargs='+',
                    help='You can give one value or a list with as many values as there are '
                    'foreground classes. Connected components with a volume of less then this/these'
                    ' constant(s) are being discraded as a part of the preprocessing. Choose '
                    'constants in the same unit as the images spacing, typically mm^3.')
parser.add_argument('--prev_stages', required=False, default=[], nargs='+',
                    help='Name the data_name, preprocessed_name and model_name of arbritraily '
                    'many previous stages here to use them as an input for model cascades.')
args = parser.parse_args()

# input arguments
apply_resizing = not args.dont_apply_resizing
if apply_resizing and args.target_spacing is not None:
    target_spacing = [float(ts) for ts in args.target_spacing]
    assert len(target_spacing) == 3, 'you must give exactly 3 floats as a target spacing'
else:
    target_spacing = None

if args.pooling_stride is not None:
    pooling_stride = [int(ps) for ps in args.pooling_stride]
    assert len(pooling_stride) == 3, 'you must give exactly 3 integers as a pooling stride'
    apply_pooling = True
else:
    pooling_stride = None
    apply_pooling = False

apply_windowing = not args.dont_apply_windowing
if apply_windowing and args.window is not None:
    window = [float(w) for w in args.window]
    assert len(window) == 2, 'you must give exactyle 2 floats as a window'
else:
    window = None

if args.scaling is not None:
    scaling = [float(s) for s in args.scaling]
    assert len(scaling) == 2, 'you must give exactly 2 floats for the input scaling'
else:
    scaling = None

if args.lb_classes is not None:
    lb_classes = [int(lbc) for lbc in args.lb_classes]
else:
    lb_classes = None

if args.lb_min_vol is not None:
    lb_min_vol = [float(lmv) for lmv in args.lb_min_vol]
else:
    lb_min_vol = None

save_only_fg_scans = not args.save_scans_without_fg

if len(args.prev_stages) % 3 != 0:
    raise ValueError('The arguments given in previous stages must be divisible by three.'
                     'The Input shold be likedata_name1, preprocessed_name1, model_name1, ...., '
                     'data_namek, preprocessed_namek, model_namek')

n_stages = len(args.prev_stages) // 3
prev_stages = []
for i in range(n_stages):
    prev_stages.append({'data_name': args.prev_stages[3*i],
                        'preprocessed_name': args.prev_stages[3*i+1],
                        'model_name': args.prev_stages[3*i+2]})


preprocessing = SegmentationPreprocessing(apply_resizing=apply_resizing,
                                          apply_pooling=apply_pooling,
                                          apply_windowing=apply_windowing,
                                          target_spacing=target_spacing,
                                          pooling_stride=pooling_stride,
                                          window=window,
                                          scaling=scaling,
                                          lb_classes=lb_classes,
                                          reduce_lb_to_single_class=args.reduce_lb_to_single_class,
                                          lb_min_vol=lb_min_vol,
                                          prev_stages=prev_stages,
                                          save_only_fg_scans=save_only_fg_scans)

preprocessing.plan_preprocessing_raw_data(args.raw_data,
                                          force_planning=True)

preprocessing.preprocess_raw_data(raw_data=args.raw_data,
                                  preprocessed_name=args.preprocessed_name,
                                  data_name=args.data_name,
                                  save_as_fp16=not args.save_as_fp32)

print('Done! Here are the preprocessing paramters:')
if args.data_name is None:
    data_name = '_'.join(sorted(args.raw_data))

# root folder of all saved preprocessed data
path_to_file = join(environ['OV_DATA_BASE'], 'preprocessed', data_name, args.preprocessed_name,
                    'preprocessing_parameters.txt')

with open(path_to_file, 'r') as file:
    print(file.read())
