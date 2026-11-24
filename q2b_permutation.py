import argparse
import os
import numpy as np
import torch
from common import (SEED, REPO_DIR, load_cifar10_arrays, get_split_indices,
                    train_classifier, flat_float, image_float, permuted_pixels)
from models import MLP, CNN

EPOCHS = 25


def get_pixel_permutation():
    path = os.path.join(REPO_DIR, "split", "pixel_permutation.npy")
    if os.path.exists(path):
        return np.load(path)
    perm = np.random.RandomState(SEED + 999).permutation(3072)
    os.makedirs(os.path.join(REPO_DIR, "split"), exist_ok=True)
    np.save(path, perm)
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlpPerm", "cnnPerm"])
    ap.add_argument("--seconds", type=int, default=None)
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    tr_x = train_x[train_idx]
    tr_y = train_y[train_idx]
    va_x = train_x[val_idx]
    va_y = train_y[val_idx]
    perm = get_pixel_permutation()

    def permuted_flat(x, epoch=0, batch=0):
        return flat_float(permuted_pixels(x, perm))

    def permuted_image(x, epoch=0, batch=0):
        return image_float(permuted_pixels(x, perm))

    torch.manual_seed(SEED)
    if args.model == "mlpPerm":
        model = MLP(3072, 500, 500)
        transform = permuted_flat
        eval_transform = permuted_flat
    else:
        model = CNN(3, 760)
        transform = permuted_image
        eval_transform = permuted_image
    train_classifier("q2b_" + args.model, model, tr_x, tr_y, va_x, va_y, transform, EPOCHS,
                     eval_transform=eval_transform, seconds=args.seconds,
                     test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    main()
