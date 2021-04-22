from ovseg.utils.io import load_pkl
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.ModelBase import ModelBase
from os import environ, listdir
from os.path import join, isdir, exists
import torch
from ovseg.utils.torch_np_utils import check_type


class SegmentationEnsemble(ModelBase):
    '''
    Ensembling Model that is used to add over softmax outputs before applying the argmax
    It is always called in inference mode!
    '''

    def __init__(self, data_name: str, model_name: str, preprocessed_name=None, val_fold=None,
                 network_name='network', fmt_write='{:.4f}',
                 model_parameters_name='model_parameters', check_if_training_is_done=True):
        self.model_cv_path = join(environ['OV_DATA_BASE'],
                                  'trained_models',
                                  data_name,
                                  model_name)
        self.check_if_training_is_done = check_if_training_is_done
        if val_fold is None:
            fold_folders = [f for f in listdir(self.model_cv_path)
                            if isdir(join(self.model_cv_path, f)) and f.startswith('fold')]
            val_fold = [int(f.split('_')[-1]) for f in fold_folders]
        super().__init__(val_fold=val_fold, data_name=data_name, model_name=model_name,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name, is_inference_only=True,
                         fmt_write=fmt_write, model_parameters_name=model_parameters_name)

        # maybe check if all the models have finished training
        if self.check_if_training_is_done:
            num_epochs = self.model_parameters['training']['num_epochs']
            not_finished_folds = []
            for fold in self.val_fold:
                path_to_attr = join(self.model_cv_path,
                                    'fold_'+str(fold),
                                    'attribute_checkpoint.pkl')
                if not exists(path_to_attr):
                    raise FileNotFoundError('Trying to check if the training is done for all folds,'
                                            ' but not checkpoint was found for fold '+str(fold)+'.')

                attr = load_pkl(path_to_attr)

                if attr['epochs_done'] != num_epochs:
                    not_finished_folds.append(fold)

            assert len(not_finished_folds) == 0, "It seems like the folds "+str(not_finished_folds)+" have not finished training."

        # create all models
        self.models = []
        for fold in self.val_fold:
            print('Creating model from fold: '+str(fold))
            model = SegmentationModel(val_fold=fold,
                                      data_name=self.data_name,
                                      model_name=self.model_name,
                                      model_parameters=self.model_parameters,
                                      preprocessed_name=self.preprocessed_name,
                                      network_name=self.network_name,
                                      is_inference_only=True,
                                      fmt_write=self.fmt_write,
                                      model_parameters_name=self.model_parameters_name
                                      )
            self.models.append(model)

        # change in evaluation mode
        for model in self.models:
            model.network.eval()

        # now we do a hack by initialising the two objects like this...
        self.preprocessing = self.models[0].preprocessing
        self.postprocessing = self.models[0].postprocessing

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

    def __call__(self, data_tpl):
        im = data_tpl['image']
        is_np,  _ = check_type(im)
        if is_np:
            im = torch.from_numpy(im).to(self.dev)
        else:
            im = im.to(self.dev)

        # the preprocessing will only do something if the image is not preprocessed yet
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            im = self.preprocessing(data_tpl, preprocess_only_im=True)

        # now the importat part: the actual enembling of sliding window evaluations
        pred = torch.stack([model.prediction(im) for model in self.models]).mean(0)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key)

        return data_tpl[self.pred_key]

    def save_prediction(self, data_tpl, folder_name, filename=None):

        self.models[0].save_prediction(data_tpl, folder_name, filename)

    def plot_prediction(self, data_tpl, folder_name, filename=None, image_key='image'):

        self.models[0].plot_prediction(data_tpl, folder_name, filename, image_key)

    def compute_error_metrics(self, data_tpl):
        return self.models[0].compute_error_metrics(data_tpl)

    def _init_global_metrics(self):
        self.models[0]._init_global_metrics()

    def _update_global_metrics(self, data_tpl):
        self.models[0]._update_global_metrics(data_tpl)
