import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from common import REPO_DIR, SEED

SUBSET_SIZE = 10000
REPRESENTATION = "q3a_ae_c128"


def main():
    z = np.load(REPO_DIR + "/data/features_%s.npz" % REPRESENTATION)
    train_f = z["train"]
    train_y = z["train_y"]
    subset_idx, _ = train_test_split(
        np.arange(len(train_f)), train_size=SUBSET_SIZE, stratify=train_y, random_state=SEED)
    scaler = StandardScaler().fit(train_f[subset_idx])
    np.savez(REPO_DIR + "/data/q4_fixed.npz",
             train_x=scaler.transform(train_f[subset_idx]).astype(np.float32),
             train_y=train_y[subset_idx],
             subset_idx=subset_idx,
             val_x=scaler.transform(z["val"]).astype(np.float32),
             val_y=z["val_y"],
             test_x=scaler.transform(z["test"]).astype(np.float32),
             test_y=z["test_y"])
    meta = {"representation": REPRESENTATION, "subset_size": SUBSET_SIZE,
            "subset_seed": SEED, "scaler": "StandardScaler fit on the 10k subset",
            "subset_class_counts": np.bincount(train_y[subset_idx]).tolist()}
    json.dump(meta, open(REPO_DIR + "/results/q4_setup.json", "w"), indent=2)
    print(meta)


if __name__ == "__main__":
    main()
