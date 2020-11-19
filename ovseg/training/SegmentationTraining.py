from ovseg.training.SingleModuleTraining import SingleModuleTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss


class SegmentationTraining(SingleModuleTraining):

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def _compute_loss(self, data_tpl):
        yb = data_tpl['label']
        out = self.module(data_tpl['image'])
        return self.loss_fctn(out, yb)
