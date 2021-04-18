import argparse

parser = argparse.ArgumentParser()
parser.add_argument("raw_data", nargs='+')
parser.add_argument("preprocessed_name")

args = parser.parse_args()

print(args.raw_data)

print(args.preprocessed_name)