from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.training.JoinedRestSegTraining import JoinedRestSegTraining
from ovseg.data.JoinedRestSegData import JoinedRestSegData
import argparse
import os

# %% parse the experiment
parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

fbp_folders = ['fbps_quater', 'fbps_full']

loss_weight = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0][args.exp]

for fbp_folder in fbp_folders:
    # %% create the data
    dose = fbp_folder.split('_')[1]
    val_fold=5
    preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')
    keys = ['image', 'label', 'fbp']
    folders = ['images', 'labels', fbp_folder]
    trn_dl_params = {'batch_size': 12, 'epoch_len': 250}
    val_dl_params = {'batch_size': 12, 'epoch_len': 25}
    
    data = JoinedRestSegData(val_fold=val_fold, keys=keys, folders=folders,
                             preprocessed_path=preprocessed_path,
                             trn_dl_params=trn_dl_params,
                             val_dl_params=val_dl_params)
    
    
    # %% some training stuff
    joint_folder = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04',
                                'pod_2d', 'joint_rest_seg_{}_{}'.format(dose, loss_weight))
    opt_params = {'lr': 0.0001}
    opt_name = 'ADAM'
    #%%
    for i in range(3):
        # here we load the pretrained restauration
        rest_model = RestaurationModel(val_fold=0,
                                       data_name='OV04',
                                       model_name='restauration_'+fbp_folder,
                                       preprocessed_name='pod_2d',
                                       dont_store_data_in_ram=True)
        
        
        
        model_params = get_model_params_2d_segmentation()
        model_params['network']['norm'] = 'inst'
        model_name = 'delete_me_2d_joint_'+dose
        
        # this is just creating a new blank segmentation model with random weights
        seg_model = SegmentationModel(val_fold=5+i,
                                      data_name='OV04',
                                      model_name=model_name,
                                      model_parameters=model_params,
                                      preprocessed_name='pod_2d',
                                      dont_store_data_in_ram=True)
    
        model_path = os.path.join(joint_folder, 'fold_{}'.format(5+i))
        training = JoinedRestSegTraining(model1=rest_model,
                                         model2=seg_model,
                                         trn_dl=data.trn_dl,
                                         model_path=model_path,
                                         loss_weight=loss_weight,
                                         val_dl=data.val_dl,
                                         opt1_params=opt_params,
                                         opt2_params=opt_params,
                                         opt1_name=opt_name,
                                         opt2_name=opt_name)
        training.train()
    
        try:
            # now validate the trained models
            results_rest = {}
            results_seg = {}
            for j in range(len(data.val_ds)):
                # load the data
                data_tpl = data.val_ds[j]
                # compute the restauration
                rest = rest_model(data_tpl)
                # evaluate and compute the error metric of the results
                results_rest[data_tpl['scan']] = rest_model.compute_error_metrics(data_tpl)
                
                # little cheat, replace the ground truht image with the restauration, the segmentation
                # model will use it for input of the prediction
                data_tpl['image'] = rest
                # eval segmentation model
                pred = seg_model(data_tpl)
                # compute DSC etc.
                results_seg[data_tpl['scan']] = seg_model.compute_error_metrics(data_tpl)
        
            rest_model._save_results_to_pkl_and_txt(results_rest, model_path, 'validation')
            seg_model._save_results_to_pkl_and_txt(results_seg, model_path, 'validation')
        except:
            print('Ups! The validation didn\'t work out :(')
