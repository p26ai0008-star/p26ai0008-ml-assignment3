import json
import numpy as np
import torch
import torch.nn.functional as F
from common import (SEED, REPO_DIR, load_cifar10_arrays, make_batches, image_float)
from models import CNN, ConvAutoencoder

SIGMA = 0.25


def fixed_noisy(x, seed_offset):
    rs = np.random.RandomState(SEED + seed_offset)
    noise = torch.from_numpy(rs.normal(0.0, SIGMA, (len(x), 32, 32, 3))).float()
    t = image_float(x)
    return torch.clamp(t + noise.permute(0, 3, 1, 2), 0.0, 1.0)


def accuracy(model, x_tensor, y, batch=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch):
            out = model(x_tensor[i:i + batch])
            correct += int((out.argmax(dim=1).numpy() == y[i:i + batch]).sum())
    return correct / len(x_tensor)


def mse(a, b, batch=1000):
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(a), batch):
            total += F.mse_loss(a[i:i + batch], b[i:i + batch], reduction="sum").item()
    return total / (len(a) * 3072)


def main():
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    torch.manual_seed(0)
    cnn = CNN(3, 760)
    cnn.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q2a_cnn_weights.pt", weights_only=True))
    clean_ae = ConvAutoencoder(128)
    clean_ae.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q3a_ae_c128_weights.pt", weights_only=True))
    dae = ConvAutoencoder(128)
    dae.load_state_dict(torch.load(REPO_DIR + "/checkpoints/q3b_dae_c128_weights.pt", weights_only=True))

    clean_test = image_float(test_x)
    noisy_test = fixed_noisy(test_x, 888001)
    clean_ae_recon = clean_ae(noisy_test)
    dae_recon = dae(noisy_test)

    results = {}
    results["acc_clean_input"] = accuracy(cnn, clean_test, test_y)
    results["acc_corrupted_input"] = accuracy(cnn, noisy_test, test_y)
    results["acc_clean_ae_recon"] = accuracy(cnn, clean_ae_recon, test_y)
    results["acc_dae_recon"] = accuracy(cnn, dae_recon, test_y)
    results["mse_clean_ae_denoise"] = mse(clean_ae_recon, clean_test)
    results["mse_dae_denoise"] = mse(dae_recon, clean_test)
    results["mse_noisy_vs_clean"] = mse(noisy_test, clean_test)
    results["sigma"] = SIGMA
    with open(REPO_DIR + "/results/q3b_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(k, round(v, 6))


if __name__ == "__main__":
    main()
