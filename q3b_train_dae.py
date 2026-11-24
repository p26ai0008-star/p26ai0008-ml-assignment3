import argparse
import numpy as np
import torch
from common import (SEED, REPO_DIR, load_cifar10_arrays, get_split_indices,
                    train_autoencoder, image_float)
from models import ConvAutoencoder

EPOCHS = 25
SIGMA = 0.25


def fixed_noise_input(x, seed_offset):
    rs = np.random.RandomState(SEED + seed_offset)
    noise = torch.from_numpy(rs.normal(0.0, SIGMA, (len(x), 32, 32, 3))).float()
    t = image_float(x)
    return torch.clamp(t + noise.permute(0, 3, 1, 2), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None)
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    va_x = train_x[val_idx]
    val_in = fixed_noise_input(va_x, 777001)
    val_target = image_float(va_x)
    test_in = fixed_noise_input(test_x, 888001)
    test_target = image_float(test_x)
    torch.manual_seed(SEED)
    model = ConvAutoencoder(128)
    train_autoencoder("q3b_dae_c128", model, train_x[train_idx], val_in, val_target,
                      lambda x, epoch, batch: fixed_noise_input(x, 600000 + epoch * 1000 + batch),
                      EPOCHS, seconds=args.seconds, test_in=test_in, test_target=test_target)
    import json
    r = json.load(open(REPO_DIR + "/results/q3b_dae_c128.json"))
    r["sigma"] = SIGMA
    json.dump(r, open(REPO_DIR + "/results/q3b_dae_c128.json", "w"), indent=2)


if __name__ == "__main__":
    main()
