import torch
import numpy as np
from ovseg.model.SegmentationModel import SegmentationModel
from time import perf_counter
from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction
import matplotlib.pyplot as plt


def dice(s, p):
    return 200 * np.sum(p*s)/np.sum(p+s)


# %%
model = SegmentationModel(0, 'all', 'warm_start_no_gamma')

# %%
data = model.data.val_ds[0]

# %%
t1 = perf_counter()
pred_model = model.predict(data)
t2 = perf_counter()
print('model {:.4f}'.format(t2-t1))
print(dice(pred_model, data['label']))
# %%
prediction = SlidingWindowPrediction(model.network, [512, 512],
                                     TTA=model.gpu_augmentation.augmentations[1])

# %%
t1 = perf_counter()
pred_new = np.argmax(prediction(data['image']), 0)
t2 = perf_counter()
print('simple: {:.4f}'.format(t2-t1))
print(dice(pred_new, pred_model))
# %%
t1 = perf_counter()
pred_flip = np.argmax(prediction(data['image'], 'flip'), 0)
t2 = perf_counter()
print('flipping: {:.4f}'.format(t2-t1))
print(dice(pred_new, pred_flip))

# %%
t1 = perf_counter()
pred_TTA = np.argmax(prediction(data['image'], 'TTA'), 0)
t2 = perf_counter()
print('TTA: {:.4f}'.format(t2-t1))
print(dice(pred_new, pred_TTA))
print(dice(pred_new, data['label']))
print(dice(data['label'], pred_flip))
print(dice(data['label'], pred_TTA))

# %% This is just randomly here! We plot the results from the other stages
lb = data['label']
contains = np.where(np.sum(lb, (0, 1)))[0]
z = np.random.choice(contains)
im_sl = torch.from_numpy(data['image'][np.newaxis, np.newaxis, :512, :512, z])
out_list = model.network(im_sl.cuda())
plt.subplot(2, 3, 1)
plt.imshow(data['label'][:512, :512, z], cmap='gray')
f = lambda x: torch.nn.functional.softmax(x, 1)
for i in range(2, 7):
    plt.subplot(2, 3, i)
    plt.imshow(f(out_list[i-2]).detach().cpu().numpy()[0, 1], cmap='gray')
