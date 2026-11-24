import argparse
import json
import os
import time
import numpy as np
from sklearn.svm import SVC
from common import REPO_DIR
from q4a_linear_primal import solve_linear_primal, linear_scores

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KERNELS = ["linear", "rbf", "poly"]
FIXED = os.path.join(REPO_DIR, "data", "q4_fixed.npz")
SWEEP_DIR = os.path.join(REPO_DIR, "results", "q4a_sweep")


def setting_path(kernel, C):
    return os.path.join(SWEEP_DIR, "%s_C%g.json" % (kernel, C))


def run_setting(kernel, C, train_x, train_y, val_x, val_y):
    path = setting_path(kernel, C)
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    if kernel == "linear":
        W, b = solve_linear_primal(train_x, train_y, C)
        dec = linear_scores(train_x, W, b)
        onehot = np.zeros_like(dec)
        onehot[np.arange(len(train_y)), train_y] = 1.0
        n_sv = int(((onehot * dec) <= 1.0 + 1e-6).any(axis=1).sum())
        train_acc = float((dec.argmax(axis=1) == train_y).mean())
        val_acc = float((linear_scores(val_x, W, b).argmax(axis=1) == val_y).mean())
        sv_fraction = n_sv / len(train_y)
        result = {"kernel": kernel, "C": C, "train_acc": train_acc, "val_acc": val_acc,
                  "sv_fraction": float(sv_fraction), "n_support_unique": int(n_sv),
                  "fit_seconds": float(time.time() - t0), "solver": "l-bfgs-primal-l2"}
        json.dump(result, open(path, "w"), indent=2)
        print("%s C=%g train %.4f val %.4f svf %.4f (%.0fs)" %
              (kernel, C, train_acc, val_acc, sv_fraction, time.time() - t0), flush=True)
        return result
    svm = SVC(kernel=kernel, C=C, cache_size=500)
    svm.fit(train_x, train_y)
    n_sv = int(len(svm.support_))
    fit_seconds = time.time() - t0
    train_acc = svm.score(train_x, train_y)
    val_acc = svm.score(val_x, val_y)
    sv_fraction = n_sv / len(train_y)
    result = {"kernel": kernel, "C": C, "train_acc": float(train_acc), "val_acc": float(val_acc),
              "sv_fraction": float(sv_fraction), "n_support_unique": int(n_sv),
              "fit_seconds": float(fit_seconds), "solver": "libsvm"}
    json.dump(result, open(path, "w"), indent=2)
    print("%s C=%g train %.4f val %.4f svf %.4f (%.0fs)" %
          (kernel, C, train_acc, val_acc, sv_fraction, fit_seconds), flush=True)
    return result


def load_fixed():
    z = np.load(FIXED)
    return (z["train_x"].astype(np.float64), z["train_y"], z["test_x"].astype(np.float64),
            z["test_y"], z["val_x"].astype(np.float64), z["val_y"])


def final_evaluation(train_x, train_y, val_x, val_y, test_x, test_y):
    final_path = os.path.join(REPO_DIR, "results", "q4a_svm_final.json")
    if os.path.exists(final_path):
        return json.load(open(final_path))
    rows = [json.load(open(p)) for p in
            [setting_path(k, c) for k in KERNELS for c in C_VALUES] if os.path.exists(p)]
    best = max(rows, key=lambda r: (r["val_acc"], -r["C"]))
    if best["kernel"] == "linear":
        W, b = solve_linear_primal(train_x, train_y, best["C"])
        train_dec = linear_scores(train_x, W, b)
        val_dec = linear_scores(val_x, W, b)
        test_dec = linear_scores(test_x, W, b)
        train_acc = float((train_dec.argmax(axis=1) == train_y).mean())
        val_acc = float((val_dec.argmax(axis=1) == val_y).mean())
        test_acc = float((test_dec.argmax(axis=1) == test_y).mean())
        preds = test_dec.argmax(axis=1)
        onehot = np.zeros_like(train_dec)
        onehot[np.arange(len(train_y)), train_y] = 1.0
        n_sv = int(((onehot * train_dec) <= 1.0 + 1e-6).any(axis=1).sum())
        sv_fraction = float(n_sv / len(train_y))
    else:
        svm = SVC(kernel=best["kernel"], C=best["C"], cache_size=500)
        svm.fit(train_x, train_y)
        train_acc = svm.score(train_x, train_y)
        val_acc = svm.score(val_x, val_y)
        test_acc = svm.score(test_x, test_y)
        preds = svm.predict(test_x)
        sv_fraction = float(len(svm.support_) / len(train_y))
    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(test_y, preds):
        confusion[t, p] += 1
    np.save(os.path.join(REPO_DIR, "results", "q4a_test_predictions.npy"), preds)
    np.save(os.path.join(REPO_DIR, "results", "q4a_test_confusion.npy"), confusion)
    final = {"kernel": best["kernel"], "C": best["C"], "subset_size": int(len(train_y)),
             "train_acc": float(train_acc), "val_acc": float(val_acc), "test_acc": float(test_acc),
             "sv_fraction": sv_fraction}
    json.dump(final, open(final_path, "w"), indent=2)
    print("FINAL", final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None)
    ap.add_argument("--kernel", choices=KERNELS, default=None)
    args = ap.parse_args()
    os.makedirs(SWEEP_DIR, exist_ok=True)
    train_x, train_y, test_x, test_y, val_x, val_y = load_fixed()
    kernels = [args.kernel] if args.kernel else KERNELS
    t0 = time.time()
    for kernel in kernels:
        for C in C_VALUES:
            if os.path.exists(setting_path(kernel, C)):
                continue
            if args.seconds is not None and time.time() - t0 > args.seconds:
                print("TIME BUDGET REACHED")
                return
            run_setting(kernel, C, train_x, train_y, val_x, val_y)
    n_done = sum(os.path.exists(setting_path(k, c)) for k in KERNELS for c in C_VALUES)
    print("SETTINGS DONE", n_done, "of", len(KERNELS) * len(C_VALUES))
    if n_done == len(KERNELS) * len(C_VALUES):
        final_evaluation(train_x, train_y, val_x, val_y, test_x, test_y)


if __name__ == "__main__":
    main()
