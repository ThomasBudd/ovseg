import torch
from time import perf_counter
from ovseg.networks.nfUNet import nfUNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", required=False, default=False, action='store_true')
parser.add_argument("--downsampling", required=False, default="0")
parser.add_argument("-b", "--use_bottleneck", required=False, default=False, action='store_true')
parser.add_argument("-f", "--fat", required=False, default=False, action='store_true')
parser.add_argument('-ps', '--patch_size', nargs='+')
args = parser.parse_args()

if args.debug:
    n_reps = 10
    filters = 8
else:
    filters = 32
    n_reps = 50


def benchmark(net, xb):
    for _ in range(10):
        out = net(xb)
        loss = (out[0]**2).mean()
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
        loss = (out[0]**2).mean()
        loss.backward()
        t += perf_counter()
        net.zero_grad()
    t /= n_reps
    print('Backwards: {:.5} seconds ellapsed'.format(t))
    torch.cuda.empty_cache()




if __name__ == '__main__':
    if args.downsampling == "0":
        kernel_sizes = [(1, 3, 3), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    elif args.downsampling == "1":
        kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    elif args.downsampling == "2":
        kernel_sizes = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    if args.debug:
        batch_size = 1
    else:
        batch_size = 2
    if args.fat:
        n_blocks = [3, 4, 6, 4]
        kernel_sizes = kernel_sizes[:4]
    else:
        n_blocks = None

    patch_size = [int(p) for p in args.patch_size]
    # print('Single precision:')
    xb = torch.randn([batch_size, 1, *patch_size], device='cuda')
    net = nfUNet(1, 2, kernel_sizes, is_2d=False, filters=filters, n_blocks=n_blocks,
                 use_bottleneck=args.use_bottleneck).cuda()
    if args.debug:
        benchmark(net, xb)
    else:
        with torch.cuda.amp.autocast():
            benchmark(net, xb)