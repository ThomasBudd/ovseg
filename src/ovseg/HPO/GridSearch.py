from ovseg.HPO.SHA import SHA
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.utils.io import load_pkl
from os import environ, listdir, makedirs
from os.path import join, exists
from time import sleep
import numpy as np
import sys
import time
import copy


class GridSearch(SHA):
    
    def __init__(self,
                 data_name: str,
                 preprocessed_name: str,
                 i_process: int,
                 parameter_names: list,
                 parameter_grids: list,
                 target_metrics: list,
                 validation_set_name: str,
                 default_model_params: dict,
                 vfs: list=[5,6,7],
                 hpo_name=None,
                 n_processes: int=8,
                 model_class=SegmentationModel,
                 ensemble_class=SegmentationEnsemble):
        self.data_name = data_name
        self.preprocessed_name = preprocessed_name
        self.i_process = i_process
        self.parameter_names = parameter_names
        self.parameter_grids = parameter_grids
        self.target_metrics = target_metrics
        self.vfs = vfs
        self.validation_set_name = validation_set_name
        self.default_model_params = default_model_params
        self.hpo_name = hpo_name
        self.n_processes = n_processes
        self.model_class = model_class
        self.ensemble_class = ensemble_class


        assert len(parameter_names) == len(parameter_grids), 'number of parameters and grids don\'t match!'
        
        self.n_stages = len(parameter_grids)
        
        # path where all models will be found
        self.path_to_models = join(environ['OV_DATA_BASE'],
                                   'trained_models',
                                   self.data_name,
                                   self.preprocessed_name)
        # we will also store some protocol there
        self.hpo_pref = 'gridsearch'
        self.hpo_log = join(self.path_to_models,
                            '_'.join([self.hpo_pref,
                                      'log',
                                      self.hpo_name,
                                      str(self.i_process)])+'.txt')

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
                                             model_parameters=model_params)
                    
                    self.print_and_log('Training '+model_name+' fold '+str(vf)+'.',1)
                    model.training.train()        
                    if self.validation_set_name == 'validation':
                        model.eval_validation_set()
                    model.clean()
            
            # wait until all other processes are done
            self.wait_until_stage_finishes()
                        
            # evaluate the ensembles if we're not doing cross validation
            self.evaluate_ensembles()

            self.wait_until_ensembles_finished()            
            self.print_stage_scores()
            # now we can update the stage
            self.stage += 1
        
        # publish the results
        self.print_final_results()

    def get_params_names_and_vfs(self):
        
        if self.stage == self.n_stages:
            return
        
        param_grid = self.parameter_grids[self.stage]
        param_name = self.parameter_names[self.stage]
        
        # list of all parameter indices and validation folds
        # accross all processes
        ind_vf_list = self.list_kronecker([list(range(len(param_grid))),
                                           self.vfs])
        self.print_and_log('Building parameters for first stage models:')
        self.print_and_log('Stage 0: {} models'.format(len(ind_vf_list)), 1)
    
        # print parameter grids
        self.print_and_log('Parameter grids: ')
        name_str = '->'.join(param_name)
        val_str = ', '.join(['{:.3e}'.format(val) for val in param_grid])
        self.print_and_log('\t' + name_str + ': '+val_str)
            
        # pick indices and folds from this process
        ind_vf_list = ind_vf_list[self.i_process::self.n_processes]
        
        if self.stage == 0:
            # in the first stage we pick the default model parameters
            model_params = self.default_model_params 
        
        else:
            # in other stages we load the previously best parameters 
            prev_pref = '_'.join([self.hpo_pref,
                                  self.hpo_name,
                                  str(self.stage-1)])
            prev_models = [mn for mn in listdir(self.path_to_models)
                           if mn.startswith(prev_pref)]
            # get all scores
            
            prev_scores = [np.nanmean(self.get_model_scores(model_name))
                           for model_name in prev_models]
            
            # get best scores
            scores_and_names = sorted(zip(prev_scores, prev_models), reverse=True)
            
            best_score, best_model_name = scores_and_names[0]
            
            model_params = load_pkl(join(self.path_to_models,
                                         best_model_name,
                                         'model_parameters.pkl'))

            self.print_and_log('Evaluated best models from previous stage:')
            
            key_str = '->'.join(self.parameter_names[self.stage-1])
            
            value_str = '{:.3e}'.format(self.nested_get(model_params,
                                                        self.parameter_names[self.stage-1]))
                
            param_str = key_str + ' = ' + val_str
                
            self.print_and_log(best_model_name+':: '+param_str+' score: {:.2f}'.format(best_score))
        
        params_list, names_list, vfs_list = [], [], []
        
        self.print_and_log('Models in this process: ')
        for ind, vf in ind_vf_list:
            
            # make model parameters
            params = copy.deepcopy(model_params)
            
            self.nested_set(params, param_name, param_grid[ind])
            
            params_list.append(params)
            
            model_name = '_'.join([self.hpo_pref,
                                   self.hpo_name,
                                   str(self.stage),
                                   '{:03d}'.format(ind)])
            names_list.append(model_name)
            
            vfs_list.append(vf)
            
            value_str = '{:.3e}'.format(param_grid[ind])
            
            key_str = '->'.join(param_name)
            
            self.print_and_log(model_name+', fold '+str(vf))
            self.print_and_log('\t'+key_str+' = '+value_str)
        
        self.print_and_log('Starting training...', 2)
        
        return zip(params_list, names_list, vfs_list)
            
    def evaluate_ensembles(self):
        
        if self.validation_set_name == 'validation':
            return
        
        # all model_names from the current stage
        model_names = ['_'.join([self.hpo_pref,
                                 self.hpo_name,
                                 str(self.stage),
                                 '{:03d}'.format(ind)]) for ind in range(len(self.parameter_grids[self.stage]))]

        # model_names in this process
        model_names = model_names[self.i_process::self.n_processes]
        
        for model_name in model_names:
            
            self.print_and_log('Evaluate ensemble '+model_name)
            
            ens = self.ensemble_class(val_fold=self.vfs,
                                      data_name=self.data_name,
                                      preprocessed_name=self.preprocessed_name,
                                      model_name=model_name)
            
            ens.wait_until_all_folds_complete()
            
            ens.eval_raw_dataset(self.validation_set_name)
    
    def get_model_scores(self, model_name):
        
        if self.validation_set_name == 'validation':
            # pick cross validation results
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'validation_CV_results.pkl')
        else:
            # pick ensemble results
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'ensemble_'+'_'.join([str(vf) for vf in self.vfs]),
                                   self.validation_set_name+'_results.pkl')
        
        # load from .pkl
        results = load_pkl(path_to_results)
        
        scores = []
        for metric in self.target_metrics:
            scores.append([results[scan][metric] for scan in results])

        scores = np.array(scores)

        return scores

    def training_finished(self, model_name, fold):
        
        path_to_trn_attr = join(self.path_to_models,
                                model_name,
                                'fold_'+str(fold),
                                'attribute_checkpoint.pkl')
        
        if not exists(path_to_trn_attr):
            return False
        
        trn_attr = load_pkl(path_to_trn_attr)
        
        if trn_attr['epochs_done'] < trn_attr['num_epochs']:
            return False
        
        if self.validation_set_name == 'validation':
            # check if the results file is at that position
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'fold_'+str(fold),
                                   'validation_results.pkl')
            
            return exists(path_to_results)
    
        return True

    def wait_until_stage_finishes(self):
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith(self.hpo_pref+'_'+self.hpo_name)]
        n_models = len(self.parameter_grids[self.stage]) * len(self.vfs)
        pref = '_'.join([self.hpo_pref, self.hpo_name, str(self.stage)])
        
        stage_finished = False
        
        while not stage_finished:
            models_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
                
                for fold in self.vfs:
                    
                    if self.training_finished(model_name, fold):
                        models_finished += 1
            
            stage_finished = models_finished == n_models
            
            if stage_finished:
                print('Stage {} finished!'.format(self.stage))
            else:
                print('{} out of {} models finished.'.format(models_finished,
                                                             n_models))
                sleep(60)          

    def wait_until_ensembles_finished(self):
        
        if self.validation_set_name == 'validation':
            return
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith(self.hpo_pref+'_'+self.hpo_name)]
        n_ensembles = len(self.parameter_grids[self.stage])
        pref = '_'.join([self.hpo_pref, self.hpo_name, str(self.stage)])
        
        ensembles_finished = 0
        
        while ensembles_finished < n_ensembles:
            ensembles_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
        
                path_to_results = join(self.path_to_models,
                                       model_name,
                                       'ensemble_'+'_'.join([str(vf) for vf in self.vfs]),
                                       self.validation_set_name+'_results.pkl')
                
                ensembles_finished += int(exists(path_to_results))
                
                
                print('{} out of {} models finished.'.format(ensembles_finished,
                                                             n_ensembles))
                
                if ensembles_finished < n_ensembles:
                    sleep(60)

    def get_current_stage(self):
        
        if not exists(self.path_to_models):
            makedirs(self.path_to_models)
            return 0
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith(self.hpo_pref+'_'+self.hpo_name)]
        
        for s in range(self.n_stages):
            
            n_models = len(self.parameter_grids[s]) * len(self.vfs)
            pref = '_'.join([self.hpo_pref, self.hpo_name, str(s)])
            
            n_models_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
                
                for fold in self.vfs:
                    
                    if self.training_finished(model_name, fold):
                        n_models_finished += 1
                    else:
                        print('Current stage: '+str(s))
                        return s 
            
            if n_models_finished < n_models:
                
                print('Current stage: '+str(s))
                return s
            
            if not self.evaluation_finished(model_name):
                print('Current stage: '+str(s))
                return s
        
        print('GridSearch finished')
        return self.n_stages
    
    def evaluation_finished(self, model_name):
        
        if self.validation_set_name == 'validation':
            raise NotImplementedError('Have to implement cross validation case')
        else:
            
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'ensemble_'+'_'.join([str(f) for f in self.vfs]),
                                   self.validation_set_name+'_results.pkl')
            return exists(path_to_results)
    
    def print_info(self):
        
        self.print_and_log('GridSearch information:')
        self.print_and_log('Validation folds: '+', '.join([str(vf) for vf in self.vfs]))
        
        for s in range(self.n_stages):
            
            n_models = len(self.parameter_grids[s]) * len(self.vfs)
            self.print_and_log('Stage {}: n_models {}'.format(s, n_models))
            self.print_and_log('Parameter: '+'->'.join(self.parameter_names[s]))
            self.print_and_log('Grid: '+ ', '.join(['{:.3e}'.format(val) for val in self.parameter_grids[s]]),
                               2)
    
    def print_stage_scores(self):
        
        param_grid = self.parameter_grids[self.stage]
        pref = '_'.join([self.hpo_pref, self.hpo_name, str(self.stage)])
        models = [pref + '_' +'{:03d}'.format(ind) for ind in range(len(param_grid))]
        
        for model_name in models:
            
            scores = self.get_model_scores(model_name)
            total_score = np.nanmean(scores)
            self.print_and_log(model_name+'. Total score: {:.3f}'.format(total_score))
            
            if len(self.target_metrics) > 1:
                metric_scores = np.nanmean(scores, 1)
                
                metric_str = ', '.join([metr+': {:.3f}'.format(score)
                                        for metr, score in zip(self.target_metrics, metric_scores)])
            
                self.print_and_log('\t '+metric_str)
            
            self.print_and_log('')