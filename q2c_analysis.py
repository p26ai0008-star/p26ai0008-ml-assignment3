import json
import numpy as np
import torch
import torch.nn.functional as F
from common import (REPO_DIR, load_cifar10_arrays, make_batches, image_float,
                    grayscale_image, replicate_gray)
from models import CNN

BINS = np.linspace(0.0, 1.0, 21)


def collect(model, x, y, transform):
    model.eval()
    confs = []
    correct_flags = []
    preds = []
    with torch.no_grad():
        for xb, yb in make_batches(x, y, 1000, 0, shuffle=False):
            out = model(transform(xb, 0, 0))
            probs = F.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            confs.append(conf.numpy())
            correct_flags.append((pred.numpy() == yb))
            preds.append(pred.numpy())
    confs = np.concatenate(confs)
    correct_flags = np.concatenate(correct_flags)
    preds = np.concatenate(preds)
    return confs, correct_flags, preds


def summarise(confs, correct_flags, preds, y):
    per_class = []
    for c in range(10):
        m = y == c
        per_class.append(float((preds[m] == c).mean()))
    hist, _ = np.histogram(confs, bins=BINS)
    return {
        "acc": float(correct_flags.mean()),
        "per_class_acc": per_class,
        "mean_conf_correct": float(confs[correct_flags].mean()),
        "mean_conf_incorrect": float(confs[~correct_flags].mean()),
        "frac_high_conf_wrong": float(((confs > 0.9) & (~correct_flags)).sum() / len(confs)),
        "conf_hist": [int(v) for v in hist],
    }


def main():
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    torch.manual_seed(0)
    cnn_rgb = CNN(3, 760)
    cnn_gray = CNN(1, 760)
    cnn_rgb.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q2a_cnn_weights.pt", weights_only=True))
    cnn_gray.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q2c_cnnGray_weights.pt", weights_only=True))

    results = {}
    confs, flags, preds = collect(cnn_rgb, test_x, test_y, image_float)
    results["rgb_on_rgb"] = summarise(confs, flags, preds, test_y)
    confs, flags, preds = collect(cnn_rgb, test_x, test_y, replicate_gray)
    results["rgb_on_gray"] = summarise(confs, flags, preds, test_y)
    confs, flags, preds = collect(cnn_gray, test_x, test_y, grayscale_image)
    results["gray_on_gray"] = summarise(confs, flags, preds, test_y)

    with open(REPO_DIR + "/results/q2c_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(k, "acc %.4f conf_correct %.4f conf_incorrect %.4f" %
              (v["acc"], v["mean_conf_correct"], v["mean_conf_incorrect"]))


if __name__ == "__main__":
    main()
