from ovseg.training.JoinedTraining import JoinedTraining
import torch


class JoinedRestSegTraining(JoinedTraining):
    
    def _eval_data_tpl(self, batch):
        # changed naming conventions, data_tpl is batch now
        fbp, im, seg = batch
        fbp, im, seg = fbp.to(self.dev), im.to(self.dev), seg.to(self.dev)
        rest = self.model1.network(fbp)
        
        # get the clean image aka restauration and the segmentation together
        batch = torch.cat([rest, seg], 1)

        # augment
        batch = self.model2.training.augmentation(batch)
        # print('batch_aug device: '+str(batch.device))
        rest_aug, seg_aug = batch[:, :-1], batch[:, -1:]
        pred = self.model2.network(rest_aug)
        # print('pred device: '+str(pred.device))
        return rest, pred, im, seg_aug
