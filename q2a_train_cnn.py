import argparse
import numpy as np
import torch
from common import (SEED, REPO_DIR, load_cifar10_arrays, get_split_indices,
                    train_classifier, image_float)
from models import CNN

EPOCHS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None)
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    tr_x = train_x[train_idx]
    tr_y = train_y[train_idx]
    va_x = train_x[val_idx]
    va_y = train_y[val_idx]
    torch.manual_seed(SEED)
    model = CNN(3, 760)
    train_classifier("q2a_cnn", model, tr_x, tr_y, va_x, va_y, image_float, EPOCHS,
                     eval_transform=image_float, seconds=args.seconds,
                     test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    main()
