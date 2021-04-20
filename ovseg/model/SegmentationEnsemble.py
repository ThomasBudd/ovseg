from ovseg.utils.io import load_pkl
from ovesg.model.SegmentationModel import SegmentationModel
from os import environ, listdir
from os.path import join, isdir


class SegmentationEnsemble(SegmentationModel):
    '''
    Ensembling Model that is used to add over softmax outputs before applying the argmax
    It is always called in inference mode!
    '''

    def __init__(self, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None, val_fold=None,
                 network_name='network',
                 fmt_write='{:.4f}', model_parameters_name='model_parameters'):
        self.model_cv_path = join(environ['OV_DATA_BASE'],
                                  'trained_models',
                                  data_name,
                                  model_name)
        if val_fold is None:
            fold_folders = [f for f in listdir(self.model_cv_path)
                            if isdir(join(self.model_cv_path, f)) and f.startswith('fold')]
            val_fold = [int(f.split('_')[-1]) for f in fold_folders]
        super().__init__(val_fold=val_fold, data_name=data_name, model_name=model_name,
                         model_parameters=model_parameters, preprocessed_name=preprocessed_name,
                         network_name=network_name, is_inference_only=True,
                         fmt_write=fmt_write, model_parameters_name=model_parameters_name)

        # to be continued

    def initialise_preprocessing(self):
        return

    def initialise_augmentation(self):
        return

    def initialise_network(self):
        return

    def initialise_postprocessing(self):
        return

    def initialise_data(self):
        return

    def initialise_training(self):
        return


