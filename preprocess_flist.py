import os
import argparse
from pathlib import Path

from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="./filelists/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=Path, default="./dataset/vctk-16k", help="path to source dir")
    args = parser.parse_args()

    idx = 0

    args.source_dir: Path = args.source_dir

    e = next(args.source_dir.iterdir())
    f = list(map(lambda x: x.name[4:], e.glob("*.wav")))
    shuffle(f)
    le = len(f)
    tr_le = int(0.9*le)
    te_le = int(0.95*le)
    train = f[:tr_le]
    test = f[tr_le:te_le]
    val = f[te_le:]

    train_e = []
    for sample in train:
        train_e += list(args.source_dir.rglob(f"*{sample}"))

    with open(args.train_list, "w") as f:
        for fname in train_e:
            f.write(str(fname) + "\n")
    
    test_e = []
    for sample in test:
        test_e += list(args.source_dir.rglob(f"*{sample}"))

    with open(args.test_list, "w") as f:
        for fname in test_e:
            f.write(str(fname) + "\n")
    
    val_e = []
    for sample in val:
        val_e += list(args.source_dir.rglob(f"*{sample}"))

    with open(args.val_list, "w") as f:
        for fname in val_e:
            f.write(str(fname) + "\n")
