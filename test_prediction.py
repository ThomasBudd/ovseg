from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction
from ovseg.networks.UNet import UNet
import numpy as np

net_2d = UNet(1, 2, [3, 3, 3], filters=8, is_2d=True)
net_3d = UNet(1, 2, [3, 3, 3], filters=8, is_2d=False)
prediction_2d = SlidingWindowPrediction(net_2d, [256, 256])
prediction_3d = SlidingWindowPrediction(net_3d, [48, 192, 192])

im = np.random.randn(1, 87, 256, 256)

# %%
pred_2d = prediction_2d(im)
pred_3d = prediction_3d(im)
