import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
args = parser.parse_args()

# 读取 pkl
with open(args.src, "rb") as f:
    data = pickle.load(f)

# 保存为 npy
np.save(args.dst, data)

print(f"Converted {args.src} -> {args.dst}")
