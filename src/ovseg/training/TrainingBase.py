import numpy as np
import os
from os.path import join, exists
import time
import pickle
import sys
from ovseg.utils.path_utils import maybe_create_path


class TrainingBase():
    '''
    Basic class for Trainer. Inherit this one for your training needs.
    Overload all classes like

    def function(...):
        super().function(...)

    '''

    def __init__(self, trn_dl, num_epochs, model_path):

        # overwrite defaults
        self.trn_dl = trn_dl
        self.num_epochs = num_epochs
        self.model_path = model_path

        # some training stuff
        self.epochs_done = 0
        self.trn_start_time = -1
        self.trn_end_time = -1
        self.total_train_time = 0

        # these attributes will be stored and recovered, append this list
        # with the attributes you want to save
        self.checkpoint_attributes = ['epochs_done', 'trn_start_time', 'trn_end_time',
                                      'total_train_time', 'num_epochs']
        self.print_attributes = ['model_path', 'num_epochs']
        # make model_path and training_log
        maybe_create_path(self.model_path)
        self.training_log = join(self.model_path, 'training_log.txt')

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
        mode = 'a' if exists(self.training_log) else 'w'
        with open(self.training_log, mode) as log_file:
            log_file.write(s)
            for _ in range(n_newlines):
                log_file.write('\n')

    def train(self):
        '''
        Basic training function where everything is happening
        '''

        if self.epochs_done >= self.num_epochs:
            # do nothing if the traing was already finished
            print('Training was already completed. Doing nothing here.')
            return
        else:
            # if we've stopped the training before by setting stop_training = True
            # this resumes it
            self.stop_training = False
        
        self.on_training_start()

        # we're using the stop_training flag to easily allow early stopping in the training
        while not self.stop_training:

            self.on_epoch_start()

            for step, batch in enumerate(self.trn_dl):

                self.do_trn_step(batch, step)

            self.on_epoch_end()
            # we save the checkpoint after calling on_epoch_end so that
            # computations added to on_epoch_end will be saved as well
            self.save_checkpoint()
            self.print_and_log('', 1)

        if self.epochs_done >= self.num_epochs:
            self.on_training_end()

    def on_training_start(self):

        self.print_and_log('Start training.', 2)
        sys.stdout.flush()

        if self.trn_start_time == -1:
            # keep date when training started
            self.trn_start_time = time.asctime()

        if self.epochs_done == 0:
            # print some training infos
            self.print_and_log('Training parameters:', 1)
            for key in self.print_attributes:
                item = self
                try:
                    for k in key.split('.'):
                        item = item.__getattribute__(k)
                        self.print_and_log(str(key)+': '+str(item))
                except AttributeError:
                    self.print_and_log(str(key)+': ERROR, item not found.')
            self.print_and_log('', 1)

    def on_epoch_start(self):

        self.print_and_log('Epoch:%d' % self.epochs_done)
        self.epoch_start_time = time.time()

    def do_trn_step(self, data_tpl, step):

        raise NotImplementedError('\'do_trn_step\' must be overloaded in the child class.\n It has '
                                  'a data tpl as input and performs one optimisation step.')

    def on_epoch_end(self):
        '''
        Basic function on what we\'re doing after each epoch.
        Add e.g. printing of training error or computations of validation error here
        '''
        epoch_time = time.time() - self.epoch_start_time
        self.total_train_time += epoch_time

        self.print_and_log('Epoch training {} done after {:.2f} seconds'
                           .format(self.epochs_done, epoch_time))
        self.epochs_done += 1
        self.print_and_log('Average epoch training time: {:.2f} seconds'
                           .format(self.total_train_time/self.epochs_done))
        if self.epochs_done >= self.num_epochs:
            self.stop_training = True

    def on_training_end(self):

        self.print_and_log('Training finished!')
        if self.trn_end_time == -1:
            self.trn_end_time = time.asctime()
        self.print_and_log('Training time: {} - {} ({:.3f}h)'.format(self.trn_start_time,
                                                                     self.trn_end_time,
                                                                     self.total_train_time/3600))
        self.save_checkpoint()

    def save_checkpoint(self, path=None):
        '''
        Saves attributes of this trainer class as .pkl file for
        later restoring
        '''
        if path is None:
            path = self.model_path
        attribute_dict = {}

        for key in self.checkpoint_attributes:

            item = self.__getattribute__(key)
            attribute_dict.update({key: item})

        with open(os.path.join(path, 'attribute_checkpoint.pkl'), 'wb') as outfile:
            pickle.dump(attribute_dict, outfile)

        self.print_and_log('Training attributes saved')

    def load_last_checkpoint(self, path=None):
        '''
        Loads trainers checkpoint, if added any custom attributes that are not
        of type scalar, tuple, list or np.ndarray.
        Overload for attributes of other types.
        '''
        if path is None:
            path = self.model_path
        path_to_trainer_checkpoint = join(path, 'attribute_checkpoint.pkl')
        if exists(path_to_trainer_checkpoint):
            with open(path_to_trainer_checkpoint, 'rb') as pickle_file:
                attribute_dict = pickle.load(pickle_file)

            for key in self.checkpoint_attributes:
                try:
                    item = attribute_dict[key]
                    self.__setattr__(key, item)
                except KeyError:
                    print('key {} was not found in loaded checkpoint. Skipping!'.format(key))
            return True
        else:
            return False
