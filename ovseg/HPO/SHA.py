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

class SHA(object):
    
    def __init__(self,
                 data_name: str,
                 preprocessed_name: str,
                 i_process: int,
                 parameter_names: list,
                 parameter_grids: list,
                 target_metrics: list,
                 validation_set_name: str,
                 default_model_params: dict,
                 n_epochs_per_stage: list=[250, 500, 1000, 1000],
                 vfs_per_stage: list=[[5], [5], [5], [6, 7]],
                 hpo_name=None,
                 n_processes: int=8,
                 n_models_per_stage=None,
                 model_class=SegmentationModel,
                 ensemble_class=SegmentationEnsemble,
                 max_wait_ensemble=3600):
        self.data_name = data_name
        self.preprocessed_name = preprocessed_name
        self.i_process = i_process
        self.parameter_names = parameter_names
        self.parameter_grids = parameter_grids
        self.target_metrics = target_metrics
        self.vfs_per_stage = vfs_per_stage
        self.validation_set_name = validation_set_name
        self.default_model_params = default_model_params
        self.n_epochs_per_stage = n_epochs_per_stage
        self.hpo_name = hpo_name
        self.n_processes = n_processes
        self.n_models_per_stage = n_models_per_stage
        self.model_class = model_class
        self.ensemble_class = ensemble_class
        self.max_wait_ensemble = max_wait_ensemble
        
        assert len(parameter_names) == len(parameter_grids), 'nber of parameters and grids don\'t match!'
        assert len(vfs_per_stage) == len(n_epochs_per_stage), 'nber of stages not consistent'
        self.n_stages = len(vfs_per_stage)
        
        for vfs in self.vfs_per_stage[:-1]:
            if len(vfs) != 1:
                raise NotImplementedError('The current implementation assumes '
                                          'that excatly one fold is trained '
                                          'per model_parameter except for the '
                                          'last stage.')
        
        # compute list of all hyper-parameters combinations
        self.parameter_combinations = self.list_kronecker(self.parameter_grids)
        self.n_combinations  = len(self.parameter_combinations)
        
        # default: number of models per stage halves at each stage
        if self.n_models_per_stage is None:
            self.n_models_per_stage = [self.n_combinations//(2**s) * len(vfs)
                                       for s, vfs in enumerate(self.vfs_per_stage)]
        
        if not self.n_models_per_stage[0] == self.n_combinations * len(self.vfs_per_stage[0]):
            raise ValueError('Number of parameter combinations {} times folds {} '
                             'and models in first stage {} don\'t match.'.format(self.n_combinations,
                                                                                 len(self.vf_per_stage[0]),
                                                                                 self.n_models_per_stage[0]))
        
        # path where all models will be found
        self.path_to_models = join(environ['OV_DATA_BASE'],
                                   'trained_models',
                                   self.data_name,
                                   self.preprocessed_name)
        # we will also store some protocol there
        self.hpo_pref = 'sha'
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
                    self.print_and_log('Evaluate '+self.validation_set_name,1)  
                    if self.validation_set_name == 'validation':
                        model.eval_validation_set()
                    else:
                        model.eval_raw_dataset(self.validation_set_name)
                    model.clean()
            
            # wait until all other processes are done
            self.wait_until_stage_finishes()
            
            # now we can update the stage
            self.stage += 1
        
        # evaluate the models as ensembles
        self.evaluate_ensembles()
        
        # publish the results
        self.print_final_results()

    def get_params_names_and_vfs(self):
        
        if self.stage == 0:
            # list of all parameter indices and validation folds
            # accross all processes
            ind_vf_list = self.list_kronecker([list(range(self.n_combinations)), 
                                               self.vfs_per_stage[0]])
            self.print_and_log('Building parameters for first stage models:')
            self.print_and_log('Stage 0: {} models'.format(self.n_combinations), 1)
        
            # print parameter grids
            self.print_and_log('Parameters grids: ')
            for name, grid in zip(self.parameter_names, self.parameter_grids):
                name_str = '->'.join(name)
                val_str = ', '.join(['{:.3e}'.format(val) for val in grid])
                self.print_and_log('\t' + name_str + ': '+val_str)
                
            # pick indices and folds from this process
            ind_vf_list = ind_vf_list[self.i_process::self.n_processes]
        
            num_epochs = self.n_epochs_per_stage[0]
            params_list, names_list, vfs_list = [], [], []
            
            self.print_and_log('Models in this process: ')
            for ind, vf in ind_vf_list:
                
                # make model parameters
                params = copy.deepcopy(self.default_model_params)
                params['training']['num_epochs'] = num_epochs
                parameter_values = self.parameter_combinations[ind]
                
                for name, value in zip(self.parameter_names,
                                       parameter_values):
                    self.nested_set(params, name, value)
                
                params_list.append(params)
                
                model_name = '_'.join([self.hpo_pref,
                                       self.hpo_name,
                                       str(num_epochs),
                                       '{:03d}'.format(ind)])
                names_list.append(model_name)
                
                vfs_list.append(vf)
                
                values = ['{:.3e}'.format(val) for val in parameter_values]
                
                keys = ['->'.join(names) for names in self.parameter_names]
                
                param_str = ', '.join([key+': '+val for key, val in zip(keys, values)])
                
                self.print_and_log('\t'+model_name+', fold '+str(vf))
                self.print_and_log(param_str)
            
            self.print_and_log('Starting training...', 2)
            
            return zip(params_list, names_list, vfs_list)
        else:
            s = self.stage
            
            # get model names and vfs from the previous stage
            prev_n_epochs = self.n_epochs_per_stage[s-1]
            prev_vfs = self.vfs_per_stage[s-1]
            prev_pref = '_'.join([self.hpo_pref,
                                  self.hpo_name,
                                  str(prev_n_epochs)])
            prev_models = [mn for mn in listdir(self.path_to_models)
                           if mn.startswith(prev_pref)]
            prev_models_vfs = self.list_kronecker([prev_models, prev_vfs])
            # get all scores
            
            prev_scores = [self.get_model_score(model_name, vf)
                           for model_name, vf in prev_models_vfs]
            
            # get best scores
            scores_and_names = sorted(zip(prev_scores, prev_models), reverse=True)
            vfs = self.vfs_per_stage[s]
            n_best = self.n_models_per_stage[s] // len(vfs)
            best_scores_and_names = scores_and_names[:n_best]
            
            # best model parameters
            num_epochs = self.n_epochs_per_stage[s]
            best_model_params = []
            
            self.print_and_log('Evaluated best models from previous stage:')
            
            for score, model_name in best_scores_and_names:
                model_params = load_pkl(join(self.path_to_models,
                                             model_name,
                                             'model_parameters.pkl'))
                # already change the num_epochs
                model_params['training']['num_epochs'] = num_epochs
                best_model_params.append(model_params)
                
                # print results with hyper parameters
                values = [self.nested_get(model_params, names) for names in self.parameter_names]
                values = ['{:.3e}'.format(val) for val in values]
                
                keys = ['->'.join(names) for names in self.parameter_names]
                
                param_str = ', '.join([key+': '+val for key, val in zip(keys, values)])
                
                self.print_and_log(model_name+':: '+param_str+' score: {:.2f}'.format(score))
            
            # now get all combinations of indices and vfs
            ind_vf_list = self.list_kronecker([list(range(len(best_model_params))), 
                                               vfs])

            self.print_and_log('Stage {}: {} models.'.format(self.stage,
                                                             len(ind_vf_list)))
            self.print_and_log('Models in this process:')
            
            # pick indices and folds from this process
            ind_vf_list = ind_vf_list[self.i_process::self.n_processes]
            params_list, names_list, vfs_list = [], [], []
            
            for ind, vf in ind_vf_list:
                
                # check if we're training folds of an already
                # existing model
                if num_epochs == prev_n_epochs:
                    # if we're repeating, reuse the model name
                    # from the previous stage
                    model_name = best_scores_and_names[ind][1]
                    
                    # when no model parameters are given, the stored ones
                    # are loaded
                    params_list.append(None)
                else:
                    # create new model name
                    model_name = '_'.join([self.hpo_pref,
                                           self.hpo_name,
                                           str(num_epochs),
                                           '{:03d}'.format(ind)])
                    # take best previous parameters
                    params_list.append(best_model_params[ind])
                
                self.print_and_log('\t'+model_name+', fold '+str(vf))
                
                names_list.append(model_name)
                
                vfs_list.append(vf)
            
            self.print_and_log('Starting training...', 2)
    
            return zip(params_list, names_list, vfs_list)

    def _get_last_vfs(self):
        vfs = []
        last_num_epochs = self.n_epochs_per_stage[-1]
        for vf, n_epochs in zip(self.vfs_per_stage, self.n_epochs_per_stage):
            if n_epochs == last_num_epochs:
                vfs = vfs + vf

        return np.unique(vfs).tolist()

    def _get_final_model_names(self):
        
        # get model names and vfs from the second last stage
        prev_n_epochs = self.n_epochs_per_stage[-2]
        prev_vfs = self.vfs_per_stage[-2]
        prev_pref = '_'.join([self.hpo_pref,
                              self.hpo_name,
                              str(prev_n_epochs)])
        prev_models = [mn for mn in listdir(self.path_to_models)
                       if mn.startswith(prev_pref)]
        prev_models_vfs = self.list_kronecker([prev_models, prev_vfs])

        # get all scores        
        prev_scores = [self.get_model_score(model_name, vf)
                       for model_name, vf in prev_models_vfs]
        
        # get best scores
        scores_and_names = sorted(zip(prev_scores, prev_models), reverse=True)
        vfs = self.vfs_per_stage[-1]
        n_best = self.n_models_per_stage[-1] // len(vfs)
        best_scores_and_names = scores_and_names[:n_best]
        
        # keep only the model names
        model_names = [model_name for score, model_name in best_scores_and_names]
    
        return model_names

    def evaluate_ensembles(self):
        
        if self.validation_set_name == 'validation':
            return
        # validation folds that should be there after the last stage
        vfs = self._get_last_vfs()
        # all model_names
        model_names = self._get_final_model_names()

        # model_names in this process
        model_names = model_names[self.i_process::self.n_processes]
        
        for model_name in model_names:
            
            self.print_and_log('Evaluate ensemble '+model_name)
            
            ens = self.ensemble_class(val_fold=vfs,
                                      data_name=self.data_name,
                                      preprocessed_name=self.preprocessed_name,
                                      model_name=model_name)
            
            ens.wait_until_all_folds_complete()
            
            ens.eval_raw_dataset(self.validation_set_name)
    
    def print_final_results(self):
        
        # validation folds that should be there after the last stage
        vfs = self._get_last_vfs()
        # all model_names
        model_names = self._get_final_model_names()
        
        scores = []
        for model_name in model_names:
            
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'ensemble_'+'_'.join([str(vf) for vf in vfs]),
                                   self.validation_set_name+'_results.pkl')
            
            while not exists(path_to_results):
                print('Waiting for '+model_name+' to finish evaluation')
                sleep(60)

            scores.append(self.get_model_score(model_name, vfs))
        
        scores_and_names = sorted(zip(scores, model_names))
        
        self.print_and_log('Evaluated best model ensembles:')
        
        for score, model_name in scores_and_names:
            model_params = load_pkl(join(self.path_to_models,
                                         model_name,
                                         'model_parameters.pkl'))
            
            # print results with hyper parameters
            values = [self.nested_get(model_params, names) for names in self.parameter_names]
            values = ['{:.3f}'.format(val) for val in values]
            
            keys = ['->'.join(names) for names in self.parameter_names]
            
            param_str = ', '.join([key+': '+val for key, val in zip(keys, values)])
            
            self.print_and_log(model_name+':: '+param_str+' score: {:.2f}'.format(score))
    
    def print_info(self):
        self.print_and_log('Got {} combinations of hyper-parameters in total'.format(self.n_combinations))
        self.print_and_log('Plan for all stages:')
        for i in range(self.n_stages):
            
            n_models = self.n_models_per_stage[i]
            n_epochs = self.n_epochs_per_stage[i]
            folds = ', '.join([str(vf) for vf in self.vfs_per_stage[i]])
            self.print_and_log('Stage {}: n_models {}, n_epochs: {}, folds: {}'.format(i,
                                                                                       n_models,
                                                                                       n_epochs,
                                                                                       folds),
                               2 if i == self.n_stages-1 else 0)
    
    def training_finished(self, model_name, fold, check_results=True):
        path_to_trn_attr = join(self.path_to_models,
                                model_name,
                                'fold_'+str(fold),
                                'attribute_checkpoint.pkl')
        
        if not exists(path_to_trn_attr):
            return False
        
        trn_attr = load_pkl(path_to_trn_attr)
        
        if trn_attr['epochs_done'] < trn_attr['num_epochs']:
            return False
        
        
        if check_results:
            # check if the results file is at that position
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'fold_'+str(fold),
                                   self.validation_set_name+'_results.pkl')

        # else:
        #     # we look at the ensembling results (last stage)
        #     all_vfs = self._get_last_vfs()
            
        #     path_to_results = join(self.path_to_models,
        #                            model_name,
        #                            'ensemble_'+'_'.join([str(vf) for vf in all_vfs]),
        #                            self.validation_set_name+'_results.pkl')
        
        return exists(path_to_results)

    def get_current_stage(self):
        
        if not exists(self.path_to_models):
            makedirs(self.path_to_models)
            return 0
        
        models_found = [model_name for model_name in listdir(self.path_to_models)
                        if model_name.startswith(self.hpo_pref+'_'+self.hpo_name)]
        
        for s in range(self.n_stages):
            
            n_models = self.n_models_per_stage[s]
            n_epochs = self.n_epochs_per_stage[s]
            vfs = self.vfs_per_stage[s]
            pref = '_'.join([self.hpo_pref, self.hpo_name, str(n_epochs)])
            
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
                        if model_name.startswith(self.hpo_pref+'_'+self.hpo_name)]
        n_models = self.n_models_per_stage[self.stage]
        n_epochs = self.n_epochs_per_stage[self.stage]
        vfs = self.vfs_per_stage[self.stage]
        pref = '_'.join([self.hpo_pref, self.hpo_name, str(n_epochs)])
        
        stage_finished = False
        
        while not stage_finished:
            models_finished = 0
            stage_models = [model_name for model_name in models_found
                            if model_name.startswith(pref)]
            
            for model_name in stage_models:
                
                for fold in vfs:
                    
                    if self.training_finished(model_name, fold, self.stage < self.n_stages):
                        models_finished += 1
            
            stage_finished = models_finished == n_models
            
            if stage_finished:
                print('Stage {} finished!'.format(self.stage))
            else:
                print('{} out of {} models finished.'.format(models_finished,
                                                             n_models))
                sleep(60)

    def get_model_score(self, model_name, fold):
        
        
        if isinstance(fold, list):
            path_to_results = join(self.path_to_models,
                                   model_name,
                                   'ensemble_'+'_'.join([str(vf) for vf in fold]),
                                   self.validation_set_name+'_results.pkl')
        else:
            # check if the results file is at that position
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
        
    def nested_set(self, dic, keys, value):
        for key in keys[:-1]:
            dic = dic[key]
        dic[keys[-1]] = value
    
    def nested_get(self, dic, keys):
        for key in keys[:-1]:
            dic = dic[key]
        return dic[keys[-1]]

    def print_and_log(self, s, n_newlines=0):
        '''
        prints, flushes and writes in the training_log
        '''
        if len(s) > 0:
            print(s)
        for _ in range(n_newlines):
            print('')
        sys.stdout.flush()
        t = time.localtime()
        if len(s) > 0:
            ts = time.strftime('%Y-%m-%d %H:%M:%S: ', t)
            s = ts + s + '\n'
        else:
            s = '\n'
        mode = 'a' if exists(self.hpo_log) else 'w'
        with open(self.hpo_log, mode) as log_file:
            log_file.write(s)
            for _ in range(n_newlines):
                log_file.write('\n')
    
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
    
    
    
    