from ovseg.training.JoinedTraining import JoinedTraining
import torch


class JoinedRestSegTraining(JoinedTraining):
    
    def _eval_data_tpl(self, batch):
        # changed naming conventions, data_tpl is batch now
        fbp, im, seg = batch
        # pytorch dataloaders add this additional axis, let's remove it
        fbp, im, seg = fbp[0].to(self.dev), im[0].to(self.dev), seg[0].to(self.dev)
        rest = self.model1.network(fbp)
        
        # get the clean image aka restauration and the segmentation together
        batch = torch.cat([rest, seg], 1)

        # augment
        batch_aug = self.model2.training.augmentation(batch)
        # print('batch_aug device: '+str(batch.device))
        rest_aug, seg_aug = batch_aug[:, :-1], batch_aug[:, -1:]
        pred = self.model2.network(rest_aug)
        # print('pred device: '+str(pred.device))
        return rest, pred, im, seg_aug
