import torch

xb = torch.rand((2, 3, 10, 50, 50))

nb, nc, nz, nx, ny = xb.shape

xb2 = torch.clone(xb).permute((0, 2, 1, 3, 4)).reshape((nb*nz, nc, nx, ny))

d = torch.abs(xb[0, :, 0] - xb2[0])
