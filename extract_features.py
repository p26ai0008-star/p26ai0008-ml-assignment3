import argparse
import os
import numpy as np
import torch
from common import REPO_DIR, load_cifar10_arrays, get_split_indices, image_float
from models import ConvAutoencoder


def extract(run_name, x, batch=500):
    model = ConvAutoencoder(128)
    path = os.path.join(REPO_DIR, "checkpoints", run_name + "_weights.pt")
    bottleneck = {"q3a_ae_c4": 4, "q3a_ae_c32": 32, "q3a_ae_c128": 128,
                  "q3b_dae_c128": 128}[run_name]
    model = ConvAutoencoder(bottleneck)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z = model.encoder(image_float(x[i:i + batch]))
            feats.append(z.reshape(len(z), -1).numpy())
    return np.concatenate(feats).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    choices=["q3a_ae_c4", "q3a_ae_c32", "q3a_ae_c128", "q3b_dae_c128"])
    args = ap.parse_args()
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = get_split_indices()
    tr = extract(args.run, train_x[train_idx])
    va = extract(args.run, train_x[val_idx])
    te = extract(args.run, test_x)
    out = os.path.join(REPO_DIR, "data", "features_%s.npz" % args.run)
    np.savez(out, train=tr, val=va, test=te,
             train_y=train_y[train_idx], val_y=train_y[val_idx], test_y=test_y)
    print(args.run, "train", tr.shape, "val", va.shape, "test", te.shape)


if __name__ == "__main__":
    main()
