import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_NUMBER = "p26ai0008"
SEED = 260008

torch.set_num_threads(2)


def load_cifar10_arrays():
    cache = os.path.join(REPO_DIR, "data", "cifar10_arrays.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["train_x"], z["train_y"], z["test_x"], z["test_y"]
    import torchvision
    root = os.path.join(REPO_DIR, "data")
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(cache, train_x=train.data, train_y=np.array(train.targets),
                        test_x=test.data, test_y=np.array(test.targets))
    z = np.load(cache)
    return z["train_x"], z["train_y"], z["test_x"], z["test_y"]


def get_split_indices():
    z = np.load(os.path.join(REPO_DIR, "split", "cifar10_split.npz"))
    return z["train_idx"], z["val_idx"]


def make_batches(x, y, batch_size, epoch, shuffle=True):
    n = len(x)
    if shuffle:
        order = np.random.RandomState(SEED + epoch).permutation(n)
    else:
        order = np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


def evaluate_classifier(model, x, y, transform, batch_size=2000):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in make_batches(x, y, batch_size, 0, shuffle=False):
            out = model(transform(xb, 0, 0))
            loss = F.cross_entropy(out, torch.from_numpy(yb).long(), reduction="sum")
            total_loss += loss.item()
            correct += int((out.argmax(dim=1).numpy() == yb).sum())
    model.train()
    return correct / len(x), total_loss / len(x)


def train_classifier(run_name, model, train_x, train_y, val_x, val_y, transform, epochs,
                     batch_size=128, lr=0.001, eval_transform=None, seconds=None,
                     extra_eval=None, extra_eval_names=None, test_x=None, test_y=None):
    os.makedirs(os.path.join(REPO_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(REPO_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    ckpt_path = os.path.join(REPO_DIR, "checkpoints", run_name + ".pt")
    csv_path = os.path.join(REPO_DIR, "logs", run_name + ".csv")
    res_path = os.path.join(REPO_DIR, "results", run_name + ".json")
    if eval_transform is None:
        eval_transform = transform
    names = extra_eval_names or []
    columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"] + names
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["epoch"]
    else:
        with open(csv_path, "w") as f:
            f.write(",".join(columns) + "\n")
    if start >= epochs:
        if os.path.exists(res_path):
            r = json.load(open(res_path))
            print(f"STATUS {r['status']}", flush=True)
            return
        start = epochs - 1
    t0 = time.time()
    done = start
    train_acc = float("nan")
    val_acc = float("nan")
    for epoch in range(start, epochs):
        if seconds is not None and epoch > start and time.time() - t0 > seconds:
            break
        model.train()
        total_loss = 0.0
        total_correct = 0
        seen = 0
        for b, (xb, yb) in enumerate(make_batches(train_x, train_y, batch_size, epoch)):
            xb_t = transform(xb, epoch, b)
            yb_t = torch.from_numpy(yb).long()
            out = model(xb_t)
            loss = F.cross_entropy(out, yb_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            total_correct += int((out.argmax(dim=1).numpy() == yb).sum())
            seen += len(yb)
        train_loss = total_loss / seen
        train_acc = total_correct / seen
        val_acc, val_loss = evaluate_classifier(model, val_x, val_y, eval_transform)
        extras = extra_eval(model) if extra_eval else {}
        row = [epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"]
        row += [f"{extras[n]:.6f}" for n in names]
        with open(csv_path, "a") as f:
            f.write(",".join(str(v) for v in row) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt_path)
        done = epoch + 1
        print(f"{run_name} epoch {done}/{epochs} train_acc {train_acc:.4f} val_acc {val_acc:.4f}", flush=True)
    result = {"run_name": run_name, "status": "complete" if done >= epochs else "running",
              "epochs_done": done, "param_count": sum(p.numel() for p in model.parameters())}
    if done >= epochs:
        torch.save(model.state_dict(), os.path.join(REPO_DIR, "checkpoints", run_name + "_weights.pt"))
        result["final_val_acc"] = val_acc
        result["final_train_acc"] = train_acc
        if test_x is not None:
            test_acc, test_loss = evaluate_classifier(model, test_x, test_y, eval_transform)
            result["test_acc"] = test_acc
            print(f"{run_name} TEST_ACC {test_acc:.4f}", flush=True)
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"STATUS {result['status']}", flush=True)


def train_autoencoder(run_name, model, train_x, val_in, val_target, train_input_fn, epochs,
                     batch_size=128, lr=0.001, seconds=None, test_in=None, test_target=None,
                     extra_eval=None, extra_eval_names=None):
    os.makedirs(os.path.join(REPO_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(REPO_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    ckpt_path = os.path.join(REPO_DIR, "checkpoints", run_name + ".pt")
    csv_path = os.path.join(REPO_DIR, "logs", run_name + ".csv")
    res_path = os.path.join(REPO_DIR, "results", run_name + ".json")
    names = extra_eval_names or []
    columns = ["epoch", "train_loss", "val_loss"] + names
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["epoch"]
    else:
        with open(csv_path, "w") as f:
            f.write(",".join(columns) + "\n")
    if start >= epochs:
        if os.path.exists(res_path):
            r = json.load(open(res_path))
            print(f"STATUS {r['status']}", flush=True)
            return
        start = epochs - 1
    t0 = time.time()
    done = start
    val_loss = float("nan")
    train_loss = float("nan")
    for epoch in range(start, epochs):
        if seconds is not None and epoch > start and time.time() - t0 > seconds:
            break
        model.train()
        total = 0.0
        seen = 0
        for b, (xb, _) in enumerate(make_batches(train_x, train_x, batch_size, epoch)):
            inp = train_input_fn(xb, epoch, b)
            target = image_float(xb)
            recon = model(inp)
            loss = F.mse_loss(recon, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        train_loss = total / seen
        model.eval()
        with torch.no_grad():
            val_recon = model(val_in)
            val_loss = F.mse_loss(val_recon, val_target).item()
        extras = extra_eval(model) if extra_eval else {}
        row = [epoch + 1, f"{train_loss:.8f}", f"{val_loss:.8f}"] + [f"{extras[n]:.8f}" for n in names]
        with open(csv_path, "a") as f:
            f.write(",".join(str(v) for v in row) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt_path)
        done = epoch + 1
        print(f"{run_name} epoch {done}/{epochs} train_mse {train_loss:.6f} val_mse {val_loss:.6f}", flush=True)
    result = {"run_name": run_name, "status": "complete" if done >= epochs else "running",
              "epochs_done": done, "param_count": sum(p.numel() for p in model.parameters())}
    if done >= epochs:
        torch.save(model.state_dict(), os.path.join(REPO_DIR, "checkpoints", run_name + "_weights.pt"))
        result["final_val_mse"] = val_loss
        result["final_train_mse"] = train_loss
        if test_in is not None:
            model.eval()
            with torch.no_grad():
                test_recon = model(test_in)
                test_loss = F.mse_loss(test_recon, test_target).item()
            result["test_mse"] = test_loss
            print(f"{run_name} TEST_MSE {test_loss:.6f}", flush=True)
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"STATUS {result['status']}", flush=True)


def flat_float(x, epoch=0, batch=0):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    return t.reshape(len(x), 3072) / 255.0


def image_float(x, epoch=0, batch=0):
    t = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
    return t.permute(0, 3, 1, 2)


def pooled_flat(x, epoch=0, batch=0):
    t = image_float(x)
    p = F.avg_pool2d(t, 2)
    return p.reshape(len(x), 768)


def augmented_flat(x, epoch, batch):
    rs = np.random.RandomState(SEED + epoch * 1000 + batch)
    dy = rs.randint(-4, 5, len(x))
    dx = rs.randint(-4, 5, len(x))
    t = image_float(x)
    padded = F.pad(t, (4, 4, 4, 4))
    out = torch.empty_like(t)
    for i in range(len(x)):
        out[i] = padded[i, :, dy[i] + 4:dy[i] + 36, dx[i] + 4:dx[i] + 36]
    return out.permute(0, 2, 3, 1).reshape(len(x), 3072)


def grayscale_image(x, epoch=0, batch=0):
    t = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
    g = t.mean(dim=3)
    return g.unsqueeze(1)


def replicate_gray(x, epoch=0, batch=0):
    g = grayscale_image(x)
    return g.repeat(1, 3, 1, 1)


def permuted_pixels(x, perm):
    flat = x.reshape(len(x), 3072)
    return flat[:, perm].reshape(len(x), 32, 32, 3)


def noisy_image(x, epoch, batch, sigma=0.25):
    rs = np.random.RandomState(SEED + 500000 + epoch * 1000 + batch)
    noise = torch.from_numpy(rs.normal(0.0, sigma, (len(x), 32, 32, 3))).float()
    t = image_float(x)
    return torch.clamp(t + noise.permute(0, 3, 1, 2), 0.0, 1.0)
