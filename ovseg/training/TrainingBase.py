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
                                      'total_train_time']

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

        self.on_training_start()

        while self.epochs_done < self.num_epochs:

            self.on_epoch_start()

            for batch in self.trn_dl:

                self.do_trn_step(batch)

            self.on_epoch_end()
            # we save the checkpoint after calling on_epoch_end so that
            # computations added to on_epoch_end will be saved as well
            self.save_checkpoint()
            self.print_and_log('', 1)

        self.on_training_end()

    def on_training_start(self):

        self.print_and_log('Start training.', 2)
        sys.stdout.flush()

        if self.trn_start_time == -1:
            # keep date when training started
            self.trn_start_time = time.asctime()

    def on_epoch_start(self):

        self.print_and_log('Epoch:%d' % self.epochs_done)
        self.epoch_start_time = time.time()

    def do_trn_step(self, data_tpl):

        raise NotImplementedError('\'do_trn_step\' must be overloaded in the child class.\n It has '
                                  'a data tpl as input and performs one optimisation step.')

    def on_epoch_end(self):
        '''
        Basic function on what we\'re doing after each epoch.
        Add e.g. printing of training error or computations of validation error here
        '''
        epoch_time = time.time() - self.epoch_start_time
        self.total_train_time += epoch_time

        self.print_and_log('Epoch {} done after {:.2f} seconds'
                           .format(self.epochs_done, epoch_time))
        self.epochs_done += 1
        self.print_and_log('Average epoch time: {:.2f} seconds'
                           .format(self.total_train_time/self.epochs_done))

    def on_training_end(self):

        self.print_and_log('Training finished!')
        self.trn_end_time = time.asctime()
        self.save_checkpoint()

    def save_checkpoint(self):
        '''
        Saves attributes of this trainer class as .pkl file for
        later restoring
        '''
        attribute_dict = {}

        for key in self.checkpoint_attributes:

            item = self.__getattribute__(key)
            attribute_dict.update({key: item})

        with open(os.path.join(self.model_path, 'attribute_checkpoint.pkl'), 'wb') as outfile:
            pickle.dump(attribute_dict, outfile)

        self.print_and_log('Training attributes saved')

    def load_last_checkpoint(self):
        '''
        Loads trainers checkpoint, if added any custom attributes that are not
        of type scalar, tuple, list or np.ndarray.
        Overload for attributes of other types.
        '''
        path_to_trainer_checkpoint = join(self.model_path, 'attribute_checkpoint.pkl')
        if exists(path_to_trainer_checkpoint):
            with open(path_to_trainer_checkpoint, 'rb') as pickle_file:
                attribute_dict = pickle.load(pickle_file)

            for key in self.checkpoint_attributes:

                item = attribute_dict[key]
                self.__setattr__(key, item)
            return True
        else:
            return False
