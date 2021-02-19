import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-lw", "--loss_weights", nargs='+', required=False, default=None)
parser.add_argument("-le", "--loss_exps", nargs='+', required=False, default=None)
args = parser.parse_args()

loss_weights = []

if args.loss_weights is not None:
    loss_weights.extend([float(lw) for lw in args.loss_weights])
if args.loss_exps is not None:
    loss_weights.extend([10**float(lw) for lw in args.loss_exps])

print(loss_weights)