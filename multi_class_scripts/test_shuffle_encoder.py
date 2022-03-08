import torch
from ovseg.networks.resUNet import UNetResShuffleEncoder


net = UNetResShuffleEncoder(in_channels=1,
                            out_channels=2,
                            is_2d=False,
                            z_to_xy_ratio=8,
                            filters=4).cuda()

xb = torch.zeros((1, 1, 32, 256, 256)).cuda()
logs_list = net(xb)
