from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

data_name = 'ApolloTCGA_BARTS_OV04'
preprocessed_name = 'pod_om'
n_procs = 5

w_1 = -1.5
w_9 = -0.5

delta_list = [-2, -1, 0, 1, 2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5][args.exp::n_procs]

for delta in delta_list:
    
    w_list = [w_1 + delta, w_9 + delta]    
    model_name = f'calibrated_{w_list[0]}_{w_list[1]}'
    
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name=data_name,
                               preprocessed_name=preprocessed_name,
                               model_name=model_name)
    
    ens.eval_raw_dataset('ICON8')
    
    