from ovseg.networks.resUNet import UNetResEncoder, UNetResStemEncoder
import torch

net1 = UNetResEncoder(1, 2, False, 5/0.67, n_blocks_list=[1, 1, 2, 6, 3],
                     filters=4).cuda()

net2 = UNetResStemEncoder(1, 2, False, 5/0.67, n_blocks_list=[1, 1, 2, 6, 3],
                          filters=4).cuda()
    
# print(net1)
print(net2)
xb = torch.zeros((1, 1, 40, 320, 320)).cuda()

with torch.no_grad():
    yb1 = net1(xb)
    yb2 = net2(xb)

print([y.shape for y in yb1])
print([y.shape for y in yb2])
