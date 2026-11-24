import argparse
import os
import numpy as np
import torch
from common import (REPO_DIR, SEED, load_cifar10_arrays, get_split_indices,
                    train_classifier, evaluate_classifier, flat_float)
from models import MLP

EPOCHS = 50
NOISE_FRACTION = 0.2


def make_corrupted_labels(sub_y):
    path = os.path.join(REPO_DIR, "split", "q1b_corrupted.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["noisy_idx"], z["corrupted_y"]
    rs = np.random.RandomState(SEED + 12345)
    sub = sub_y
    n_noisy = int(len(sub) * NOISE_FRACTION)
    noisy_idx = rs.choice(len(sub), n_noisy, replace=False)
    corrupted_y = sub.copy()
    shifted = rs.randint(1, 10, n_noisy)
    corrupted_y[noisy_idx] = (sub[noisy_idx] + shifted) % 10
    os.makedirs(os.path.join(REPO_DIR, "split"), exist_ok=True)
    np.savez(path, noisy_idx=noisy_idx, corrupted_y=corrupted_y)
    return noisy_idx, corrupted_y


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
    noisy_idx, corrupted_y = make_corrupted_labels(tr_y)
    clean_mask = np.ones(len(tr_y), dtype=bool)
    clean_mask[noisy_idx] = False
    clean_idx = np.where(clean_mask)[0]
    torch.manual_seed(SEED)
    model = MLP(3072, 500, 500)

    def extra_eval(m):
        acc_clean, _ = evaluate_classifier(m, tr_x[clean_idx], corrupted_y[clean_idx], flat_float)
        acc_noisy, _ = evaluate_classifier(m, tr_x[noisy_idx], corrupted_y[noisy_idx], flat_float)
        return {"train_acc_clean": acc_clean, "train_acc_noisy": acc_noisy}

    train_classifier("q1b_mlpL_noisy", model, tr_x, corrupted_y, va_x, va_y, flat_float, EPOCHS,
                     seconds=args.seconds, extra_eval=extra_eval,
                     extra_eval_names=["train_acc_clean", "train_acc_noisy"])
    result_path = os.path.join(REPO_DIR, "results", "q1b_mlpL_noisy.json")
    import json
    r = json.load(open(result_path))
    r["noisy_fraction"] = NOISE_FRACTION
    r["n_noisy"] = int(len(noisy_idx))
    r["n_clean"] = int(len(clean_idx))
    json.dump(r, open(result_path, "w"), indent=2)


if __name__ == "__main__":
    main()
