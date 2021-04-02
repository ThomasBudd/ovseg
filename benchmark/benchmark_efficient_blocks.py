import torch
from time import perf_counter

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
            self.conv1 = torch.nn.Conv2d(32, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(32, 32, kernel_size, padding=padding)
        else:
            self.conv1 = torch.nn.Conv3d(32, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(32, 32, kernel_size, padding=padding)
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
            self.conv1 = torch.nn.Conv2d(32, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(32, 32, kernel_size, padding=padding)
            self.norm1 = torch.nn.InstanceNorm2d(32)
            self.norm2 = torch.nn.InstanceNorm2d(32)
        else:
            self.conv1 = torch.nn.Conv3d(32, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(32, 32, kernel_size, padding=padding)
            self.norm1 = torch.nn.InstanceNorm3d(32)
            self.norm2 = torch.nn.InstanceNorm3d(32)
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
            self.conv1 = torch.nn.Conv2d(32, 16, 1)
            self.conv2 = torch.nn.Conv2d(16, 16, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv2d(16, 16, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(16, 32, 1)
        else:
            self.conv1 = torch.nn.Conv3d(32, 16, 1)
            self.conv2 = torch.nn.Conv3d(16, 16, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv3d(16, 16, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(16, 32, 1)
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
            self.conv1 = torch.nn.Conv2d(32, 8, 1)
            self.conv2 = torch.nn.Conv2d(8, 8, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv2d(8, 8, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(8, 32, 1)
        else:
            self.conv1 = torch.nn.Conv3d(32, 8, 1)
            self.conv2 = torch.nn.Conv3d(8, 8, kernel_size, padding=padding)
            self.conv3 = torch.nn.Conv3d(8, 8, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(8, 32, 1)
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
            self.conv1 = torch.nn.Conv2d(6, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv2d(32, 6, 1)
            self.conv3 = torch.nn.Conv2d(6, 32, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv2d(32, 6, 1)
        else:
            self.conv1 = torch.nn.Conv3d(6, 32, kernel_size, padding=padding)
            self.conv2 = torch.nn.Conv3d(32, 6, 1)
            self.conv3 = torch.nn.Conv3d(6, 32, kernel_size, padding=padding)
            self.conv4 = torch.nn.Conv3d(32, 6, 1)
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



if __name__ == '__main__':
    kernel_sizes = [(3, 3), (3, 3, 3), (1, 3, 3)]
    patch_sizes = [(512, 512), (96, 96, 96),
                   (64, 128, 128)]
    n_channels = [32, 32, 32, 32, 6]
    for kernel_size, patch_size in zip(kernel_sizes, patch_sizes):
        print(kernel_size)
        # print('Single precision:')
        xb = torch.randn(patch_size, device='cuda')
        blocks = [b(kernel_size).cuda() for b in [Block0, Block1, Block2, Block3, Block4]]
        # for i, block in enumerate(blocks):
            # print(i)
            # benchmark(block, xb)
        print('Half precision:')
        with torch.cuda.amp.autocast():
            for i, block in enumerate(blocks):
                xb = torch.randn((2, n_channels[i], *patch_sizes[i]), device='cuda')
                print(i)
                benchmark(block, xb)
