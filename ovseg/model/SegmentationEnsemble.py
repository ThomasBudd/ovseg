from ovseg.utils.io import load_pkl
from ovesg.model.SegmentationModel import SegmentationModel


class SegmentationEnsemble(SegmentationModel):

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', model_parameters_name='model_parameters',
                 plot_n_random_slices=1, dont_store_data_in_ram=False):
        super().super().

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


