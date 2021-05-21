from ovseg.model.SegmentationModel import SegmentationModel
import matplotlib.pyplot as plt
import torch
import numpy as np

model_name = 'repeat_3d_1_prg_lrn_32_128'
data_name = 'OV04'
val_fold = 5
model = SegmentationModel(val_fold, data_name, model_name, preprocessed_name='pod_half')
self = model.network

#%%
for batch in model.data.val_dl:
    break


# %%
xb = batch[:1, :1, :, 64:192, 64:192].cuda().type(torch.float)
yb = batch[0, 1, :, 64:192, 64:192].numpy()
z = np.argmax(np.sum(yb, (1, 2)))
plt.subplot(1, 2, 1)
plt.imshow(xb.cpu().numpy()[0, 0, z], cmap='gray')
plt.contour(yb[z] > 0)

xb_list = []
up_list = []
logs_list = []
# contracting path
for block in self.blocks_down:
    xb = block(xb)
    xb_list.append(xb)

# expanding path without logits
for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
    xb = self.upconvs[i](xb)
    up_list.append(xb)
    xb = self.concats[i](xb, xb_list[i])
    xb = self.blocks_up[i](xb)

# expanding path with logits
for i in range(self.n_pyramid_scales - 1, -1, -1):
    xb = self.upconvs[i](xb)
    up_list.append(xb)
    xb = self.concats[i](xb, xb_list[i])
    xb = self.blocks_up[i](xb)
    logs = self.all_logits[i](xb)
    logs_list.append(logs)

sm = torch.nn.functional.softmax(logs, 1)
up_list = up_list[::-1]
plt.subplot(1, 2, 2)
plt.imshow(sm.detach().cpu().numpy()[0, 1, z], cmap='gray')
# %%
k = 4
s = 1
im_skip = xb_list[s][0, :, z].detach().cpu().numpy()
im_up = up_list[s][0, :, z].detach().cpu().numpy()
ch_list = np.random.choice(np.arange(im_skip.shape[0]), size=k, replace=False)
plt.figure()
for i, ch in enumerate(ch_list):
    plt.subplot(2, k, i + 1)
    plt.imshow(im_skip[ch], cmap='gray')
    plt.colorbar()
    plt.subplot(2, k, i + 1 + k)
    plt.imshow(im_up[ch], cmap='gray')
    plt.colorbar()
