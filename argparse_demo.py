import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--name', type=str, required=True)
parser.add_argument('name', type=str)

args = parser.parse_args()

print('Hello, ', args.name)