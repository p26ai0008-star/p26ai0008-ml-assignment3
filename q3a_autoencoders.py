import argparse
import torch
from common import (SEED, REPO_DIR, load_cifar10_arrays, get_split_indices,
                    train_autoencoder, image_float)
from models import ConvAutoencoder

EPOCHS = 25
BOTTLENECKS = [4, 32, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottleneck", type=int, required=True, choices=BOTTLENECKS)
    ap.add_argument("--seconds", type=int, default=None)
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    va_x = train_x[val_idx]
    te_x = test_x
    val_in = image_float(va_x)
    val_target = val_in
    test_in = image_float(te_x)
    test_target = test_in
    torch.manual_seed(SEED)
    model = ConvAutoencoder(args.bottleneck)
    run_name = "q3a_ae_c%d" % args.bottleneck
    train_autoencoder(run_name, model, train_x[train_idx], val_in, val_target, image_float,
                      EPOCHS, seconds=args.seconds, test_in=test_in, test_target=test_target)


if __name__ == "__main__":
    main()
