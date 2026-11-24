import csv
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import REPO_DIR, load_cifar10_arrays, image_float
from models import ConvAutoencoder, CNN

FIG = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150,
                     "figure.constrained_layout.use": True})


def read_log(name):
    rows = list(csv.DictReader(open(os.path.join(REPO_DIR, "logs", name + ".csv"))))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def fig_q1a():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))
    specs = [("mlpL", "MLP-L (1.79M)"), ("mlpS", "MLP-S (0.41M)"),
             ("mlpB", "MLP-B aug (1.79M)"), ("mlpP", "MLP-P pooled (1.78M)")]
    for ax, (m, title) in zip(axes.flat, specs):
        c = read_log("q1a_" + m)
        ax.plot(c["epoch"], c["train_acc"], label="train", color="#1f77b4")
        ax.plot(c["epoch"], c["val_acc"], label="val", color="#d62728")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.3, 0.75)
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "q1a_curves.pdf"))
    plt.close(fig)


def fig_q1b():
    c = read_log("q1b_mlpL_noisy")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(c["epoch"], c["train_acc_clean"], label="train, clean labels", color="#1f77b4")
    ax.plot(c["epoch"], c["train_acc_noisy"], label="train, random labels", color="#ff7f0e")
    ax.plot(c["epoch"], c["val_acc"], label="validation", color="#d62728")
    best = int(c["epoch"][np.argmax(c["val_acc"])])
    ax.axvline(best, color="gray", ls="--", lw=0.8)
    ax.annotate("val peak (ep %d)" % best, (best, 0.30), fontsize=7,
                xytext=(best + 2, 0.25), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(os.path.join(FIG, "q1b_memorisation.pdf"))
    plt.close(fig)


def fig_q2a():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
    for ax, (run, title) in zip(axes, [("q1a_mlpL", "MLP-L (1.79M params)"),
                                       ("q2a_cnn", "CNN (1.78M params)")]):
        c = read_log(run)
        ax.plot(c["epoch"], c["train_acc"], label="train", color="#1f77b4")
        ax.plot(c["epoch"], c["val_acc"], label="val", color="#d62728")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "q2a_curves.pdf"))
    plt.close(fig)


def fig_q2b():
    runs = [("MLP", "q2b_mlpPerm", "q1a_mlpL"), ("CNN", "q2b_cnnPerm", "q2a_cnn")]
    clean, perm = [], []
    for _, pc, cl in runs:
        clean.append(json.load(open(REPO_DIR + "/results/%s.json" % cl))["test_acc"])
        perm.append(json.load(open(REPO_DIR + "/results/%s.json" % pc))["test_acc"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(2)
    ax.bar(x - 0.18, clean, 0.34, label="natural order", color="#1f77b4")
    ax.bar(x + 0.18, perm, 0.34, label="permuted pixels", color="#d62728")
    for xi, v in zip(x - 0.18, clean):
        ax.text(xi, v + 0.01, "%.3f" % v, ha="center", fontsize=7)
    for xi, v in zip(x + 0.18, perm):
        ax.text(xi, v + 0.01, "%.3f" % v, ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["MLP", "CNN"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "q2b_permutation.pdf"))
    plt.close(fig)


def fig_q2c():
    r = json.load(open(REPO_DIR + "/results/q2c_analysis.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    centers = (BINS_EDGES[:-1] + BINS_EDGES[1:]) / 2
    for ax, (key, title) in zip(axes, [("rgb_on_rgb", "RGB CNN on RGB test"),
                                       ("gray_on_gray", "gray CNN on gray test")]):
        hist = np.array(r[key]["conf_hist"], dtype=float)
        ax.plot(centers, hist / hist.sum(), color="#1f77b4")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("max softmax confidence")
        ax.set_ylabel("fraction of test images")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
    fig.savefig(os.path.join(FIG, "q2c_confidence.pdf"))
    plt.close(fig)
    pc_rgb = np.array(r["rgb_on_rgb"]["per_class_acc"])
    pc_gray = np.array(r["gray_on_gray"]["per_class_acc"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(10)
    ax.bar(x - 0.18, pc_rgb, 0.34, label="RGB", color="#1f77b4")
    ax.bar(x + 0.18, pc_gray, 0.34, label="grayscale", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_ABBR, rotation=45, fontsize=6)
    ax.set_ylabel("per-class test accuracy")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "q2c_perclass.pdf"))
    plt.close(fig)


BINS_EDGES = np.linspace(0.0, 1.0, 21)
CLASS_NAMES_ABBR = ["air", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def fig_q3a():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for c, col in zip([4, 32, 128], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        log = read_log("q3a_ae_c%d" % c)
        ax.plot(log["epoch"], log["val_loss"], label="c=%d (%d units)" % (c, c * 16),
                color=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "q3a_val_mse.pdf"))
    plt.close(fig)


def strip_images(rows, cols_list, fname, widths=None):
    fig, axes = plt.subplots(len(rows), len(cols_list),
                             figsize=(1.15 * len(cols_list), 1.15 * len(rows)))
    for i, row in enumerate(rows):
        for j in range(len(cols_list)):
            ax = axes[i, j] if len(rows) > 1 else axes[j]
            img = cols_list[j](i)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(widths[j], fontsize=7)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_q3_images():
    _, _, test_x, _ = load_cifar10_arrays()
    idx = [1, 7, 11, 5003]
    torch.manual_seed(0)
    clean_ae = ConvAutoencoder(128)
    clean_ae.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q3a_ae_c128_weights.pt",
                                        weights_only=True))
    dae = ConvAutoencoder(128)
    dae.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q3b_dae_c128_weights.pt",
                                   weights_only=True))
    from q3b_eval import fixed_noisy
    sub = test_x[idx]
    noisy = fixed_noisy(sub, 424242)
    with torch.no_grad():
        rec_clean_ae = clean_ae(noisy).permute(0, 2, 3, 1).numpy()
        rec_dae = dae(noisy).permute(0, 2, 3, 1).numpy()
        rec_clean_on_clean = clean_ae(image_float(sub)).permute(0, 2, 3, 1).numpy()
    noisy_np = noisy.permute(0, 2, 3, 1).numpy()
    orig = sub.astype(np.float32) / 255.0
    rows = list(range(len(idx)))
    cols_list = [lambda i, a=orig: a[i], lambda i, a=noisy_np: a[i],
                 lambda i, a=rec_clean_ae: a[i], lambda i, a=rec_dae: a[i]]
    strip_images(rows, cols_list, "q3b_recon.pdf",
                 ["clean input", "noisy input", "clean AE recon", "DAE recon"])


def fig_q4a():
    kernels = ["linear", "rbf", "poly"]
    colors = {"linear": "#1f77b4", "rbf": "#d62728", "poly": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
    for kernel in kernels:
        cs, va, sv = [], [], []
        for C in [0.01, 0.1, 1, 10, 100, 1000]:
            p = os.path.join(REPO_DIR, "results", "q4a_sweep", "%s_C%g.json" % (kernel, C))
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            cs.append(C)
            va.append(r["val_acc"])
            sv.append(r["sv_fraction"])
        axes[0].plot(cs, va, "o-", color=colors[kernel], label=kernel)
        axes[1].plot(cs, sv, "o-", color=colors[kernel], label=kernel)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation accuracy")
    axes[1].set_ylabel("support vector fraction")
    fig.savefig(os.path.join(FIG, "q4a_sweep.pdf"))
    plt.close(fig)


def fig_q4b():
    data = json.load(open(REPO_DIR + "/results/q4b_pairs.json"))
    _, _, test_x, _ = load_cifar10_arrays()
    z = np.load(REPO_DIR + "/data/q4_fixed.npz")
    fig, axes = plt.subplots(3, 8, figsize=(6.5, 2.7))
    for r, pair in enumerate(data["pairs"]):
        a, b = pair["classes"]
        bnd = pair["boundary"]
        for j in range(8):
            ax = axes[r, j]
            ax.imshow(test_x[bnd["test_index"][j]])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("%+.3f" % bnd["score"][j], fontsize=6)
            if j == 0:
                ax.set_ylabel("%s/%s" % (CLASS_NAMES_ABBR[a], CLASS_NAMES_ABBR[b]),
                              fontsize=6)
    fig.savefig(os.path.join(FIG, "q4b_boundary.pdf"))
    plt.close(fig)


def fig_q4c():
    r = json.load(open(REPO_DIR + "/results/q4c_clustering.json"))
    table = np.array(r["contingency"])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    frac = table / table.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("true class")
    ax.set_ylabel("cluster")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES_ABBR, rotation=45, fontsize=6)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in range(table.shape[0])], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(FIG, "q4c_contingency.pdf"))
    plt.close(fig)


def main():
    fig_q1a()
    fig_q1b()
    fig_q2a()
    fig_q2b()
    fig_q2c()
    fig_q3a()
    fig_q3_images()
    fig_q4a()
    fig_q4b()
    fig_q4c()
    print("figures written:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
