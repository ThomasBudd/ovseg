import torch
from time import perf_counter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ps', '--patch_size', nargs='+')
parser.add_argument("--filters", required=False, default=32)
parser.add_argument("--batch_size", required=False, default=2)

args = parser.parse_args()
f = int(args.filters)
n_reps = 50


def benchmark(net, xb):
    for _ in range(10):
        out = net(xb)
        loss = (out**2).mean()
        loss.backward()
        net.zero_grad()
    t = 0
    for _ in range(n_reps):
        t -= perf_counter()
        net(xb)
        t += perf_counter()
    t /= n_reps
    print('Forwards:  {:.5} seconds ellapsed'.format(t))

    t = 0
    for _ in range(n_reps):
        t -= perf_counter()
        out = net(xb)
        loss = (out**2).mean()
        loss.backward()
        t += perf_counter()
        net.zero_grad()
    t /= n_reps
    print('Backwards: {:.5} seconds ellapsed'.format(t))
    torch.cuda.empty_cache()


class Block0(torch.nn.Module):
    # typical conv nonlin block
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(f, f, kernel_size, padding=padding)
        else:
            self.conv1 = torch.nn.Conv3d(f, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(f, f, kernel_size, padding=padding)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu

    def forward(self, xb):

        xb = self.nonlin1(self.conv1(xb))
        xb = self.nonlin2(self.conv2(xb))
        return xb


class Block1(torch.nn.Module):
    # typical conv norm nonlin block
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(f, f, kernel_size, padding=padding)
            self.norm1 = torch.nn.InstanceNorm2d(f)
            self.norm2 = torch.nn.InstanceNorm2d(f)
        else:
            self.conv1 = torch.nn.Conv3d(f, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(f, f, kernel_size, padding=padding)
            self.norm1 = torch.nn.InstanceNorm3d(f)
            self.norm2 = torch.nn.InstanceNorm3d(f)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu

    def forward(self, xb):

        xb = self.nonlin1(self.norm1(self.conv1(xb)))
        xb = self.nonlin2(self.norm2(self.conv2(xb)))
        return xb


class Block2(torch.nn.Module):
    # nf block without SE and weight norm
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f, f // 2, 1)
            self.conv2 = torch.nn.Conv2d(f // 2, f // 2, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv2d(f // 2, f // 2, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(f // 2, f, 1)
        else:
            self.conv1 = torch.nn.Conv3d(f, f // 2, 1)
            self.conv2 = torch.nn.Conv3d(f // 2, f // 2, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv3d(f // 2, f // 2, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(f // 2, f, 1)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu
        self.nonlin3 = torch.nn.functional.relu
        self.nonlin4 = torch.nn.functional.relu

    def forward(self, xb):

        skip = xb
        xb = self.conv1(self.nonlin1(xb))
        xb = self.conv2(self.nonlin2(xb))
        xb = self.conv3(self.nonlin3(xb))
        xb = self.conv4(self.nonlin4(xb))
        return xb + skip


class Block3(torch.nn.Module):
    # nf block without SE and weight norm
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f, f // 4, 1)
            self.conv2 = torch.nn.Conv2d(f // 4, f // 4, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv2d(f // 4, f // 4, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(f // 4, f, 1)
        else:
            self.conv1 = torch.nn.Conv3d(f, f // 4, 1)
            self.conv2 = torch.nn.Conv3d(f // 4, f // 4, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv3d(f // 4, f // 4, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(f // 4, f, 1)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu
        self.nonlin3 = torch.nn.functional.relu
        self.nonlin4 = torch.nn.functional.relu

    def forward(self, xb):

        skip = xb
        xb = self.conv1(self.nonlin1(xb))
        xb = self.conv2(self.nonlin2(xb))
        xb = self.conv3(self.nonlin3(xb))
        xb = self.conv4(self.nonlin4(xb))
        return xb + skip


class Block4(torch.nn.Module):
    # nf block without SE and weight norm
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f // 4, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(f, f // 4, 1)
            self.conv3 = torch.nn.Conv2d(f // 4, f, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(f, f // 4, 1)
        else:
            self.conv1 = torch.nn.Conv3d(f // 4, f, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(f, f // 4, 1)
            self.conv3 = torch.nn.Conv3d(f // 4, f, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(f, f // 4, 1)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu
        self.nonlin3 = torch.nn.functional.relu
        self.nonlin4 = torch.nn.functional.relu

    def forward(self, xb):

        skip = xb
        xb = self.conv1(self.nonlin1(xb))
        skip = self.conv2(self.nonlin2(xb)) + skip
        xb = self.conv3(self.nonlin3(skip))
        xb = self.conv4(self.nonlin4(xb)) + skip
        return xb


class Block5(torch.nn.Module):
    # nf block without SE and weight norm
    def __init__(self, kernel_size):
        super().__init__()
        padding = [(k-1)//2 for k in kernel_size]
        if len(kernel_size) == 2:
            self.conv1 = torch.nn.Conv2d(f // 4, f, 1)
            self.conv2 = torch.nn.Conv2d(f, f, kernel_size, padding=padding, groups=f)
            self.conv3 = torch.nn.Conv2d(f, f // 4, 1)
            self.conv4 = torch.nn.Conv2d(f // 4, f, 1)
            self.conv5 = torch.nn.Conv2d(f, f, kernel_size, padding=padding, groups=f)
            self.conv6 = torch.nn.Conv2d(f, f // 4, 1)
        else:
            self.conv1 = torch.nn.Conv3d(f // 4, f, 1)
            self.conv2 = torch.nn.Conv3d(f, f, kernel_size, padding=padding, groups=f)
            self.conv3 = torch.nn.Conv3d(f, f // 4, 1)
            self.conv4 = torch.nn.Conv3d(f // 4, f, 1)
            self.conv5 = torch.nn.Conv3d(f, f, kernel_size, padding=padding, groups=f)
            self.conv6 = torch.nn.Conv3d(f, f // 4, 1)
        self.nonlin1 = torch.nn.functional.relu
        self.nonlin2 = torch.nn.functional.relu
        self.nonlin3 = torch.nn.functional.relu
        self.nonlin4 = torch.nn.functional.relu
        self.nonlin5 = torch.nn.functional.relu
        self.nonlin6 = torch.nn.functional.relu

    def forward(self, xb):

        skip = xb
        xb = self.conv1(self.nonlin1(xb))
        xb = self.conv2(self.nonlin2(xb))
        skip = self.conv3(self.nonlin3(xb)) + skip
        xb = self.conv4(self.nonlin4(skip))
        xb = self.conv5(self.nonlin5(xb))
        xb = self.conv6(self.nonlin6(xb)) + skip
        return xb


if __name__ == '__main__':

    if len(args.patch_size) == 2:
        kernel_size = (3, 3)
    elif args.patch_size[0] < args.patch_size[1]:
        kernel_size = (1, 3, 3)
    else:
        kernel_size = (3, 3, 3)

    n_channels = [f, f, f, f, f // 4, f // 4]
    names = ['conv', 'conv_norm', 'bottleneck 2', 'bottleneck 4', 'fused MB_conv', 'MB_conv']
    # print('Single precision:')
    patch_size = [int(p) for p in args.patch_size]
    blocks = [b(kernel_size).cuda() for b in [Block0, Block1, Block2, Block3, Block4, Block5]]

    print('Half precision:')
    with torch.cuda.amp.autocast():
        for i, block in enumerate(blocks):
            xb = torch.randn((2, n_channels[i], *patch_size), device='cuda')
            print(names[i])
            benchmark(block, xb)
