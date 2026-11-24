import os
import numpy as np
from sklearn.model_selection import train_test_split
from common import REPO_DIR, SEED, load_cifar10_arrays


def main():
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_x)), test_size=5000, stratify=train_y, random_state=SEED)
    os.makedirs(os.path.join(REPO_DIR, "split"), exist_ok=True)
    np.savez(os.path.join(REPO_DIR, "split", "cifar10_split.npz"),
             train_idx=train_idx, val_idx=val_idx)
    print("seed", SEED)
    print("train size", len(train_idx), "val size", len(val_idx))
    print("val class counts", np.bincount(train_y[val_idx]))
    print("train class counts", np.bincount(train_y[train_idx]))


if __name__ == "__main__":
    main()
