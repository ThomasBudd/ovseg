import torch
from time import perf_counter
from ovseg.networks.nfUNet import nfUNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--downsampling", required=False, default="1")
parser.add_argument("--debug", required=False, default=False, action='store_true')

parser.add_argument('--upsampling', required=False, default='conv')
parser.add_argument('--is_efficient', required=False, default=False, action='store_true')
parser.add_argument("--skip_fac", required=False, default=1.0)
parser.add_argument("--align_corners", required=False, default=False, action='store_true')
parser.add_argument("--is_inference_network", required=False, default=False, action='store_true')
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
    torch.cuda.empty_cache()




if __name__ == '__main__':

    if args.downsampling == "0":
        kernel_sizes = [(1, 3, 3), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        patch_size = [28, 224, 224]
    elif args.downsampling == "1":
        kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        patch_size = [48, 192, 192]
    elif args.downsampling == "2":
        kernel_sizes = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        patch_size = [48, 96, 96]
    if args.debug:
        filters = 8
        batch_size = 1
    else:
        filters = 32
        batch_size = 2
    # print('Single precision:')
    xb = torch.randn([batch_size, 1, *patch_size], device='cuda')
    net = nfUNet(1, 2, kernel_sizes, is_2d=False, filters=filters,
                 upsampling=args.upsampling,
                 factor_skip_conn=float(args.skip_fac),
                 align_corners=args.align_corners,
                 is_inference_network=args.is_inference_network,
                 is_efficient=args.is_efficient).cuda()
    if args.debug:
        benchmark(net, xb)
    else:
        with torch.cuda.amp.autocast():
            benchmark(net, xb)
