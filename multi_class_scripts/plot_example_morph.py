import numpy as np
import torch
from ovseg.utils.torch_morph import dial_2d, eros_2d, opening_2d, closing_2d
import matplotlib.pyplot as plt

selem = np.ones((3, 3))
selem[1, :] = 1
selem[:, 1] = 1
selem = selem/selem.sum()

# %%
seg = np.zeros((1, 1, 21, 21))
seg[0, 0, 6:16, 6:16] = 1
seg[0, 0, 2:6, 10] = 1
seg[0, 0, 11:16, 10] = 0

plt.subplot(2, 3, 2)
plt.imshow(seg[0, 0], cmap='gray')
plt.axis('off')
plt.title('Segmentation')

eros = eros_2d(torch.from_numpy(seg.copy()).cuda(), selem)
plt.subplot(2, 3, 1)
plt.imshow(eros[0, 0].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('Erosion')

opening = opening_2d(torch.from_numpy(seg.copy()).cuda(), selem)
plt.subplot(2, 3, 4)
plt.imshow(opening[0, 0].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('Opening')

dial = dial_2d(torch.from_numpy(seg.copy()).cuda(), selem)
plt.subplot(2, 3, 3)
plt.imshow(dial[0, 0].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('Dilation')

closing = closing_2d(torch.from_numpy(seg.copy()).cuda(), selem)
plt.subplot(2, 3, 6)
plt.imshow(closing[0, 0].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('Closing')
