from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_effUNet
from ovseg.utils.io import load_pkl
from os import environ, listdir
from os.path import join, exists
from time import sleep
import numpy as np

class SHA(object):
    
    def __init__(self,
                 data_name: str,
                 preprocessed_name: str,
                 n_process: int,
                 parameter_names: list,
                 parameter_grids: list,
                 target_metrics: list,
                 validation_set_name: str,
                 n_epochs_per_stage: list=[250, 500, 1000, 1000],
                 vfs_per_stage: list=[[5], [5], [5], [6,7]],
                 hpo_name=None,
                 n_processes: int=8,
                 n_models_per_stage=None,
                 model_class=SegmentationModel,
                 ensemble_class=SegmentationEnsemble):
        self.data_name = data_name
        self.preprocessed_name = preprocessed_name
        self.n_process = n_process
        self.parameter_names = parameter_names
        self.parameter_grids = parameter_grids
        self.target_metrics = target_metrics
        self.vfs_per_stage = vfs_per_stage
        self.validation_set_name = validation_set_name
        self.n_epochs_per_stage = n_epochs_per_stage
        self.hpo_name = hpo_name
        self.n_processes = n_processes
        self,n_models_per_stage = n_models_per_stage
        self.model_class = model_class
        self.ensemble_class = ensemble_class
        
        assert len(parameter_names) == len(parameter_grids), 'nber of parameters and grids don\'t match!'
        assert len(vfs_per_stage) == len(n_epochs_per_stage), 'nber of stages not consistent'
        self.n_stages = len(vfs_per_stage)
        
        # compute list of all hyper-parameters combinations
        self.parameter_combinations = self.list_kronecker(self.parameter_grids)
        self.n_combinations  = len(self.parameter_combinations)
        
        # default: number of models per stage halves at each stage
        if self.n_models_per_stage is None:
            self.n_models_per_stage = [self.n_combinations//(2**i) for i in range(self.n_stages)]
        
        if not self.n_models_per_stage[0] == self.n_combinations * self.vf_per_stage[0]:
            raise ValueError('Number of parameter combinations {} times folds {} '
                             'and models in first stage {} don\'t match.'.format(self.n_combinations,
                                                                                 self.vf_per_stage[0],
                                                                                 self.n_models_per_stage[0]))
        
        
        self.path_to_models = join(environ['OV_DATA_BASE'],
                                   'trained_models',
                                   self.data_name,
                                   self.preprocessed_name)
        
        self.stage = self.get_current_stage()
        self.print_info()
    
    def launch(self):
        
        # until we're done with all stages
        while self.stage < self.n_stages:
            
            # train all models at this stage
            for model_params, model_name, vf in self.get_params_names_and_vfs():
                
                if not self.training_finished(model_name, vf):
                
                    model = self.model_class(val_fold=vf,
                                             data_name=self.data_name,
                                             preprocessed_name=self.preprocessed_name,
                                             model_name=model_name,
                                             model_paramters=model_params)
                    model.training.tain()                
                    model.eval_raw_dataset(self.validation_set_name)
                    model.clean()
            
            # wait until all other processes are done
            self.wait_until_stage_finishes()
            
            # now we can update the stage
            self.stage += 1
        
        self.print_final_results()

    def get_params_names_and_vfs(self):
        
        if self.stage == 0:
            ind_list = 
        
    
    def print_info(self):
        print('Got {} combinations of hyper-parameters in total'.format(self.n_combinations))
        print('Plan for all stages:')
        for i in range(self.n_stages):
            
            n_models = self.n_models_per_stage[i]
            n_epochs = self.n_epochs_per_stage[i]
            folds = ', '.join([str(vf) for vf in self.vfs_per_stage[i]])
            print('Stage {}: n_models {}, n_epochs: {}, folds: {}'.format(i,
                                                                            n_models,
                                                                            n_epochs,
                                                                            folds))
    
    def training_finished(self, model_name, fold, check_results=True):
        path_to_trn_attr = join(self.path_to_models,
                                model_name,
                                'fold_'+str(fold),
                                'attribute_checkpoint.pkl')
        
        if not exists(path_to_trn_attr):
            return False
        
        trn_attr = load_pkl(path_to_trn_attr)
        
        if trn_attr['epochs_done'] <= trn_attr['num_epochs']:
            return False
        
        
        if check_results:
            # check if the results file is at that position
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'fold_'+str(fold),
                                   self.validation_set_name+'_results.pkl')

        else:
            # we look at the ensembling results (last stage)
            all_vfs = []
            for vfs in self.vfs_per_stage:
                all_vfs = all_vfs + vfs
            all_vfs = np.unique(all_vfs).tolist()
            
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'ensemble_'+'_'.join([str(vf) for vf in all_vfs]),
                                   self.validation_set_name+'_results.pkl')
        
        return exists(path_to_results)

    def get_current_stage(self):
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith('hpo_'+self.hpo_name)]
        
        for s in range(self.n_stages):
            
            n_models = self.n_models_per_stage[s]
            n_epochs = self.n_epochs_per_stage[s]
            vfs = self.vfs_per_stage[s]
            pref = '_'.join(['hpo', self.hpo_name, str(n_epochs)])
            
            n_models_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
                
                for fold in vfs:
                    
                    if self.training_finished(model_name, fold, s<self.n_stages):
                        n_models_finished += 1
                    else:
                        print('Current stage: '+str(s))
                        return s
            
            if n_models_finished < n_models:
                
                print('Current stage: '+str(s))
                return s
        
        print('SHA finished')
        return self.n_stages

    def wait_until_stage_finishes(self):
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith('hpo_'+self.hpo_name)]
        n_models = self.n_models_per_stage[self.stage]
        n_epochs = self.n_epochs_per_stage[self.stage]
        vfs = self.vfs_per_stage[self.stage]
        pref = '_'.join(['hpo', self.hpo_name, str(n_epochs)])
        
        stage_finished = False
        
        while not stage_finished:
            models_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
                
                for fold in vfs:
                    
                    while not self.training_finished(model_name, fold, self.stage < self.n_stages):
                        models_finished += 1
            
            stage_finished = models_finished == n_models
            
            if stage_finished:
                print('Stage {} finished!'.format(self.stage))
            else:
                print('{} out of {} models finished.'.format(models_finished,
                                                             n_models))
                sleep(60)

    def get_model_score(self, model_name, fold):
        
        path_to_results = join(self.path_to_models,
                               model_name,
                               'fold_'+str(fold),
                               self.validation_set_name+'_results.pkl')
        
        results = load_pkl(path_to_results)
        
        scores = []
        for metric in self.target_metrics:
            scores.extend([results[scan][metric] for scan in results])
        
        return np.nanmean(scores)
            
    def list_kronecker(self, list_of_lists):
        
        kronecker_list = [[item] for item in list_of_lists[0]]
        
        for L in list_of_lists[1:]:
            new_kronecker_list = []
            for tpl in kronecker_list:
                new_kronecker_list.extend([tpl + [l] for l in L])
            kronecker_list = new_kronecker_list
        
        return kronecker_list
        
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic[key]
        dic[keys[-1]] = value
        return dic
    
    
# %%
parameter_grids = [[0.99, 0.98, 0.95, 0.9],
                   np.logspace(np.log10(3e-5), np.log10(3e-6), 8).tolist(),
                   [1,2]]

parameter_combinations = [[param] for param in parameter_grids[0]]
for param_grid in parameter_grids[1:]:
    new_combinations = []
    for tpl in parameter_combinations:
        new_combinations.extend([tpl + [param] for param in param_grid])
    parameter_combinations = new_combinations
    
    
    
    