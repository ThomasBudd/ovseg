from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.utils.io import read_data_tpl_from_nii
import numpy as np

data_tpl = read_data_tpl_from_nii('ApolloTCGA', 600)
ens = SegmentationEnsemble(data_name='OV04', preprocessed_name='pod_half',
                           model_name='pod_half_default')

lb = ens.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
prep_ens = ens(data_tpl)

print(200 * np.sum(prep_ens * lb) / np.sum(lb + prep_ens))

preps_sm = [model(data_tpl) for model in ens.models]
print([200 * np.sum(prep * lb) / np.sum(lb + prep) for prep in preps_sm])
