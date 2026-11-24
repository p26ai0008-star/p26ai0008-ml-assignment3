import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import REPO_DIR, load_cifar10_arrays, image_float
from models import ConvAutoencoder

RUNS = ["q3a_ae_c4", "q3a_ae_c32", "q3a_ae_c128", "q3b_dae_c128"]


def dae_clean_test_mse():
    _, _, test_x, _ = load_cifar10_arrays()
    model = ConvAutoencoder(128)
    model.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q3b_dae_c128_weights.pt", weights_only=True))
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(test_x), 500):
            x = image_float(test_x[i:i + 500])
            total += F.mse_loss(model(x), x, reduction="sum").item()
    return total / (len(test_x) * 3072)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, choices=RUNS)
    args = ap.parse_args()
    path = REPO_DIR + "/results/q3c_probes.json"
    results = json.load(open(path)) if __import__("os").path.exists(path) else {}
    runs = [args.run] if args.run else RUNS
    for run in runs:
        z = np.load(REPO_DIR + "/data/features_%s.npz" % run)
        scaler = StandardScaler().fit(z["train"])
        train_f = scaler.transform(z["train"]).astype(np.float32)
        test_f = scaler.transform(z["test"]).astype(np.float32)
        probe = LogisticRegression(max_iter=1000)
        probe.fit(train_f, z["train_y"])
        train_acc = probe.score(train_f, z["train_y"])
        test_acc = probe.score(test_f, z["test_y"])
        results[run] = {"dim": int(z["train"].shape[1]), "probe_train_acc": float(train_acc),
                        "probe_test_acc": float(test_acc)}
        print(run, results[run], flush=True)
        json.dump(results, open(path, "w"), indent=2)
    if "dae_clean_test_mse" not in results:
        results["dae_clean_test_mse"] = dae_clean_test_mse()
        json.dump(results, open(path, "w"), indent=2)
        print("dae_clean_test_mse", results["dae_clean_test_mse"])


if __name__ == "__main__":
    main()
