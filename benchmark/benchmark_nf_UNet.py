import torch
from time import perf_counter
from ovseg.networks.nfUNet import nfUNet_benchmark
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", required=False, default=False, action='store_true')
parser.add_argument("--downsampling", required=False, default="0")
parser.add_argument("-b", "--use_bottleneck", required=False, default=False, action='store_true')
parser.add_argument("-br", "--bottleneck_ratio", required=False, default=2)
parser.add_argument("-f", "--fat", required=False, default=False, action='store_true')
parser.add_argument('-ps', '--patch_size', nargs='+')
parser.add_argument('--upsampling', required=False, default='conv')
parser.add_argument("--skip_fac", required=False, default=0.5)
parser.add_argument("--align_corners", required=False, default=False, action='store_true')
parser.add_argument("--filters", required=False, default=32)
parser.add_argument("--batch_size", required=False, default=2)

args = parser.parse_args()

if args.debug:
    n_reps = 10
    filters = 8
else:
    filters = int(args.filters)
    n_reps = 50


def benchmark(net, xb):
    for _ in range(10):
        out = net(xb)
        loss = (out[0]**2).mean()
        loss.backward()
        net.zero_grad()
    net.eval()
    t = 0
    for _ in range(n_reps):
        t -= perf_counter()
        net(xb)
        t += perf_counter()
    t /= n_reps
    print('Forwards:  {:.5} seconds ellapsed'.format(t))
    net._print_perf_times()
    net.train()
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
    print('A full training would take {:.3f}h'.format(250000 * t / 3600))
    net._print_perf_times()
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
        batch_size = int(args.batch_size)
    if args.fat:
        n_blocks = [1, 2, 6, 3]
        kernel_sizes = kernel_sizes[:4]
    else:
        n_blocks = None

    patch_size = [int(p) for p in args.patch_size]
    # print('Single precision:')
    xb = torch.randn([batch_size, 1, *patch_size], device='cuda')
    net = nfUNet_benchmark(1, 2, kernel_sizes, is_2d=False, filters=filters, n_blocks=n_blocks,
                           use_bottleneck=args.use_bottleneck, upsampling=args.upsampling,
                           bottleneck_ratio=int(args.bottleneck_ratio),
                           factor_skip_conn=float(args.skip_fac),
                           align_corners=args.align_corners).cuda()
    if args.debug:
        benchmark(net, xb)
    else:
        with torch.cuda.amp.autocast():
            benchmark(net, xb)

    net._print_perf_times()