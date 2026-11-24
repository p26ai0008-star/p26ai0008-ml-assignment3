import argparse
import torch
from common import (SEED, load_cifar10_arrays, get_split_indices, train_classifier,
                    flat_float, augmented_flat, pooled_flat)
from models import MLP

EPOCHS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlpL", "mlpB", "mlpS", "mlpP"])
    ap.add_argument("--seconds", type=int, default=None)
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    tr_x = train_x[train_idx]
    tr_y = train_y[train_idx]
    va_x = train_x[val_idx]
    va_y = train_y[val_idx]
    torch.manual_seed(SEED)
    if args.model == "mlpL":
        model = MLP(3072, 500, 500)
        transform = flat_float
        eval_transform = flat_float
    elif args.model == "mlpB":
        model = MLP(3072, 500, 500)
        transform = augmented_flat
        eval_transform = flat_float
    elif args.model == "mlpS":
        model = MLP(3072, 128, 128)
        transform = flat_float
        eval_transform = flat_float
    else:
        model = MLP(768, 1000, 1000)
        transform = pooled_flat
        eval_transform = pooled_flat
    train_classifier("q1a_" + args.model, model, tr_x, tr_y, va_x, va_y, transform, EPOCHS,
                     eval_transform=eval_transform, seconds=args.seconds,
                     test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    main()
