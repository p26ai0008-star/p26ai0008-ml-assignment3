import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from common import REPO_DIR, SEED

K_VALUES = [10, 20, 30, 40]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    z = np.load(os.path.join(REPO_DIR, "data", "features_q3a_ae_c128.npz"))
    subset = np.load(os.path.join(REPO_DIR, "split", "q4_subset_indices.npy"))
    sub_f = z["train"][subset].astype(np.float64)
    sub_y = z["train_y"][subset]
    val_f = z["val"].astype(np.float64)
    val_y = z["val_y"]
    test_f = z["test"].astype(np.float64)
    test_y = z["test_y"]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(sub_f)
    sub_f = scaler.transform(sub_f)
    val_f = scaler.transform(val_f)
    test_f = scaler.transform(test_f)

    sel_path = os.path.join(REPO_DIR, "results", "q4c_k_selection.json")
    if not os.path.exists(sel_path):
        rows = []
        for k in K_VALUES:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            km.fit(sub_f)
            sil_train = silhouette_score(sub_f, km.labels_, sample_size=3000, random_state=SEED)
            val_labels = km.predict(val_f)
            sil_val = silhouette_score(val_f, val_labels, random_state=SEED)
            rows.append({"k": k, "silhouette_train": float(sil_train), "silhouette_val": float(sil_val)})
            print(rows[-1], flush=True)
        best = max(rows, key=lambda r: r["silhouette_val"])
        json.dump({"rows": rows, "selected_k": best["k"]}, open(sel_path, "w"), indent=2)
    sel = json.load(open(sel_path))
    k = sel["selected_k"]
    print("selected k", k)

    final_path = os.path.join(REPO_DIR, "results", "q4c_clustering.json")
    if not os.path.exists(final_path):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        km.fit(sub_f)
        test_labels = km.predict(test_f)
        ari = adjusted_rand_score(test_y, test_labels)
        nmi = normalized_mutual_info_score(test_y, test_labels)
        contingency = np.zeros((k, 10), dtype=int)
        for c, t in zip(test_labels, test_y):
            contingency[c, t] += 1
        purity = float(sum(contingency.max(axis=1)) / contingency.sum())
        result = {"selected_k": k, "test_ari": float(ari), "test_nmi": float(nmi),
                  "test_purity": purity, "contingency": contingency.tolist()}
        np.save(os.path.join(REPO_DIR, "results", "q4c_test_cluster_labels.npy"), test_labels)
        svm_final = json.load(open(os.path.join(REPO_DIR, "results", "q4a_svm_final.json")))
        result["svm_test_acc"] = svm_final["test_acc"]
        preds = np.load(os.path.join(REPO_DIR, "results", "q4a_test_predictions.npy"))
        result["svm_vs_cluster_ari"] = float(adjusted_rand_score(preds, test_labels))
        result["svm_vs_cluster_nmi"] = float(normalized_mutual_info_score(preds, test_labels))
        svm_per_class = {}
        for c in range(10):
            m = test_y == c
            svm_per_class[CLASSES[c]] = float((preds[m] == c).mean())
        result["svm_per_class_acc"] = svm_per_class
        clu_per_class = {}
        for c in range(10):
            counts = contingency[:, c]
            best_cl = counts.argmax()
            frac_in_best = counts[best_cl] / counts.sum()
            clu_per_class[CLASSES[c]] = {"dominant_cluster": int(best_cl),
                                         "fraction_in_dominant": float(frac_in_best)}
        result["cluster_per_class"] = clu_per_class
        json.dump(result, open(final_path, "w"), indent=2)
    result = json.load(open(final_path))
    print("k", result["selected_k"], "ARI %.4f NMI %.4f purity %.4f" %
          (result["test_ari"], result["test_nmi"], result["test_purity"]))
    for cls, v in result["cluster_per_class"].items():
        print(cls, "svm %.3f" % result["svm_per_class_acc"][cls],
              "dominant cluster %d frac %.3f" % (v["dominant_cluster"], v["fraction_in_dominant"]))


if __name__ == "__main__":
    main()
