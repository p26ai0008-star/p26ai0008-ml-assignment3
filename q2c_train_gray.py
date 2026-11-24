import argparse
import numpy as np
import torch
from common import (SEED, load_cifar10_arrays, get_split_indices,
                    train_classifier, grayscale_image)
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
    model = CNN(1, 760)
    train_classifier("q2c_cnnGray", model, tr_x, tr_y, va_x, va_y, grayscale_image, EPOCHS,
                     eval_transform=grayscale_image, seconds=args.seconds,
                     test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    main()
