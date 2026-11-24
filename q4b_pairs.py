import json
import os
import numpy as np
from sklearn.svm import SVC
from common import REPO_DIR

FIXED = os.path.join(REPO_DIR, "data", "q4_fixed.npz")


def main():
    z = np.load(FIXED)
    train_x = z["train_x"].astype(np.float64)
    train_y = z["train_y"]
    test_x = z["test_x"].astype(np.float64)
    test_y = z["test_y"]
    confusion = np.load(os.path.join(REPO_DIR, "results", "q4a_test_confusion.npy"))
    off = confusion + confusion.T
    np.fill_diagonal(off, 0)
    ranked = sorted(((off[a, b], a, b) for a in range(10) for b in range(a + 1, 10)),
                    reverse=True)
    selected = [(a, b) for _, a, b in ranked[:3]]
    final = json.load(open(os.path.join(REPO_DIR, "results", "q4a_svm_final.json")))
    out = {"kernel": final["kernel"], "C": final["C"],
           "selection_rule": "top 3 pairs by symmetric test confusion mass",
           "pairs": []}
    for a, b in selected:
        mask_tr = (train_y == a) | (train_y == b)
        xa = train_x[mask_tr]
        ya = train_y[mask_tr]
        svm = SVC(kernel=final["kernel"], C=final["C"], cache_size=500)
        svm.fit(xa, ya)
        sv_indices_global = mask_tr.nonzero()[0][svm.support_]
        sv_labels = ya[svm.support_]
        mask_te = (test_y == a) | (test_y == b)
        xt = test_x[mask_te]
        yt = test_y[mask_te]
        scores = svm.decision_function(xt)
        preds = svm.predict(xt)
        test_acc = float((preds == yt).mean())
        order = np.argsort(np.abs(scores))
        boundary_pos = order[:8]
        boundary = {"test_index": mask_te.nonzero()[0][boundary_pos].tolist(),
                    "score": scores[boundary_pos].tolist(),
                    "true": yt[boundary_pos].tolist(),
                    "pred": preds[boundary_pos].tolist(),
                    "correct": (preds[boundary_pos] == yt[boundary_pos]).tolist()}
        out["pairs"].append({
            "classes": [int(a), int(b)],
            "confusion_mass": int(off[a, b]),
            "test_pair_acc": test_acc,
            "n_train_pair": int(mask_tr.sum()),
            "n_support": int(len(svm.support_)),
            "sv_fraction_pair": float(len(svm.support_) / mask_tr.sum()),
            "sv_label_counts": {str(int(a)): int((sv_labels == a).sum()),
                                str(int(b)): int((sv_labels == b).sum())},
            "sv_example_indices": sv_indices_global[:8].tolist(),
            "boundary": boundary,
        })
        print("pair", a, b, "acc %.4f" % test_acc, "sv", len(svm.support_), flush=True)
    with open(os.path.join(REPO_DIR, "results", "q4b_pairs.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
