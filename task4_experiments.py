# Task 4 — a better leaf classifier, measured against one fixed baseline.
#
# Everything below is the Task 3 pipeline: the same LeafDataset, the same
# LeafNet, the same training loop (train mode -> batch loop -> eval mode on
# the validation set). Each variant changes exactly ONE thing versus the
# row it is compared with, listed in VARIANTS at the bottom, and every run
# starts from torch.manual_seed(seed) so the comparison is fair.
#
# Every variant is measured on:
#   - the standard validation set (the val folder as it is), and
#   - a camera-like validation set: the same val images once randomly
#     cropped, rotated and colour-jittered with a fixed seed (seed 7),
#     built ONCE before any training, so all variants see the same images.
# The weights of the best validation epoch are kept (copy.deepcopy of the
# state_dict, as in 1g) and measured too, in the "best" columns.
#
# The whole sweep is ~15 min on a CPU-only torch, so it is built to survive
# being stopped: every finished variant is appended to task4_results.csv and
# task4_results.txt right away, and a rerun continues where it stopped.
#   python -u task4_experiments.py            run everything (or resume)
#   python task4_experiments.py --list        show the variants with numbers
#   python task4_experiments.py --only 1 2 3  run just those variants
#   python task4_experiments.py --fresh       ignore the csv and start over

import copy
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
base_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
aug_transform = transforms.Compose([                    # Task 3d, geometric only
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aug_color_transform = transforms.Compose([              # + colour augmentation
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


class LeafDataset(Dataset):                             # Task 3a, unchanged
    def __init__(self, folder, transform):
        self.transform = transform
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            for path in sorted((folder_path / class_name).iterdir()):
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((str(path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, class_idx = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx


class LeafNet(nn.Module):                               # Task 3b, unchanged
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LeafNetReg(LeafNet):
    """LeafNet with the two architecture switches of Task 4:
    gap=True     -> global average pooling instead of flatten(1)
    dropout=p    -> nn.Dropout(p) after the hidden ReLU layer
    With gap the feature map (64, 6, 6) is averaged to (64, 1, 1), so fc1
    takes 64 inputs instead of 64 * 6 * 6."""

    def __init__(self, gap=False, dropout=0.0):
        super().__init__()                              # the same conv layers as LeafNet
        self.gap = nn.AdaptiveAvgPool2d(1) if gap else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        if gap:                                         # averaged features: 64 instead of 64 * 6 * 6
            self.fc1 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.gap(x).flatten(1) if self.gap else x.flatten(1)
        x = torch.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        return self.fc2(x)


def make_mobilenet():
    """Task 2b: pretrained mobilenet_v3_small with a new 4-output last layer."""
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


def evaluate(model, loader):
    """Validation loss and accuracy of the current weights (eval mode, no_grad)."""
    model.eval()
    loss_sum, correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            loss_sum += loss.item() * len(labels)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def run(name, make_model, transform, epochs, seed=0, optimizer="adam",
        weight_decay=0.0, label_smoothing=0.0, cosine=False, freeze_backbone=False):
    """The Task 3c/3d training loop plus the one-line options a variant needs.
    Returns the row for the summary table."""
    torch.manual_seed(seed)
    model = make_model()
    if freeze_backbone:                                 # transfer: train only the new last layer
        for p in model.features.parameters():
            p.requires_grad_(False)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_ds = LeafDataset("data/leaves/train", transform)
    val_ds = LeafDataset("data/leaves/val", base_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=32)

    start = time.perf_counter()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    print(f"\n=== {name} | seed {seed} | {epochs} epochs | {len(train_ds)} train images ===")
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if scheduler:
            scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader)
        if val_loss < best_loss:                        # keep the weights of the best epoch (1g)
            best_loss, best_epoch, best_state = val_loss, epoch + 1, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1:2d}  training loss {loss_sum / len(train_loader):.4f}  "
              f"validation accuracy {val_acc:.3f}  loss {val_loss:.4f}")
    train_time = time.perf_counter() - start

    final_val_loss, final_val_acc = evaluate(model, val_loader)
    _, final_cam_acc = evaluate(model, camera_loader)
    model.load_state_dict(best_state)                   # the best-epoch weights
    _, best_val_acc = evaluate(model, val_loader)
    _, best_cam_acc = evaluate(model, camera_loader)
    print(f"best validation loss {best_loss:.4f} in epoch {best_epoch}, trained in {train_time:.0f} s")
    return dict(name=name, epochs=epochs, train_loss=loss_sum / len(train_loader),
                val_acc=final_val_acc, cam_acc=final_cam_acc,
                best_val_acc=best_val_acc, best_cam_acc=best_cam_acc,
                best_epoch=best_epoch, time=train_time)


# The camera-like validation set: built ONCE, fixed seed 7, then identical for every variant.
camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
torch.manual_seed(7)
_val_plain = LeafDataset("data/leaves/val", base_transform)
_camera_x = torch.stack([camera_transform(Image.open(p).convert("RGB")) for p, _ in _val_plain.samples])
_camera_y = torch.tensor([y for _, y in _val_plain.samples])
camera_loader = DataLoader(TensorDataset(_camera_x, _camera_y), batch_size=32)
print(f"camera-like validation set: {len(_camera_y)} images, crop+rotation+colour, fixed seed 7")

RESULTS_CSV = Path("task4_results.csv")           # one row per FINISHED variant
FIELDS = ["name", "epochs", "train_loss", "val_acc", "cam_acc",
          "best_val_acc", "best_cam_acc", "best_epoch", "time"]


def as_text(row):
    """One row with the numbers formatted, for the csv."""
    return dict(name=row["name"], epochs=row["epochs"], train_loss=f"{row['train_loss']:.4f}",
                val_acc=f"{row['val_acc']:.4f}", cam_acc=f"{row['cam_acc']:.4f}",
                best_val_acc=f"{row['best_val_acc']:.4f}", best_cam_acc=f"{row['best_cam_acc']:.4f}",
                best_epoch=row["best_epoch"], time=f"{row['time']:.0f}")


def load_rows():
    """Rows an earlier, interrupted run already measured."""
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return [dict(name=r["name"], epochs=int(r["epochs"]), train_loss=float(r["train_loss"]),
                     val_acc=float(r["val_acc"]), cam_acc=float(r["cam_acc"]),
                     best_val_acc=float(r["best_val_acc"]), best_cam_acc=float(r["best_cam_acc"]),
                     best_epoch=int(r["best_epoch"]), time=float(r["time"]))
                for r in csv.DictReader(f)]


def summary_table(rows):
    return [(f"{'variant':52s} {'tr loss':>7s} {'val fin':>7s} {'val best':>8s} "
             f"{'cam fin':>7s} {'cam best':>8s} {'best ep':>7s} {'time s':>6s}"),
            "-" * 104] + [
        f"{r['name']:52s} {r['train_loss']:7.4f} {r['val_acc']:7.3f} {r['best_val_acc']:8.3f} "
        f"{r['cam_acc']:7.3f} {r['best_cam_acc']:8.3f} {r['best_epoch']:7d} {r['time']:6.0f}"
        for r in rows]


def save(rows):
    """Called after EVERY variant, so a run that is stopped early keeps its results."""
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([as_text(r) for r in rows])
    table = summary_table(rows)
    with open("task4_results.txt", "w", encoding="utf-8") as f:
        f.write("Task 4 — leaf classifier improvements, seed 0 unless noted, "
                "batch 32, Adam/AdamW lr 1e-3, 96x96 images\n")
        f.write("val = standard validation set, cam = camera-like set (crop+rotation+colour, fixed seed 7), "
                "best = weights of the best validation epoch\n\n")
        f.write("\n".join(table) + "\n")
    return table


VARIANTS = [
    # --- the baseline and the augmentation changes (fast, from scratch) ---
    dict(name="B0 baseline (Task 3, no augmentation)", make_model=LeafNet, transform=base_transform, epochs=15),
    dict(name="+ geometric augmentation (Task 3d)", make_model=LeafNet, transform=aug_transform, epochs=15),
    dict(name="+ colour jitter on top", make_model=LeafNet, transform=aug_color_transform, epochs=15),
    # --- one regularization change at a time, each on top of the plain baseline ---
    dict(name="+ global average pooling", make_model=lambda: LeafNetReg(gap=True), transform=base_transform, epochs=15),
    dict(name="+ dropout 0.25", make_model=lambda: LeafNetReg(dropout=0.25), transform=base_transform, epochs=15),
    dict(name="+ AdamW weight decay 1e-4", make_model=LeafNet, transform=base_transform, epochs=15,
         optimizer="adamw", weight_decay=1e-4),
    dict(name="+ label smoothing 0.1", make_model=LeafNet, transform=base_transform, epochs=15,
         label_smoothing=0.1),
    # --- transfer learning (Task 2b model) ---
    dict(name="transfer: frozen backbone, head only", make_model=make_mobilenet, transform=base_transform,
         epochs=15, freeze_backbone=True),
    dict(name="transfer: fine-tune whole network", make_model=make_mobilenet, transform=base_transform, epochs=15),
    dict(name="transfer + aug + colour + cosine, 30 ep", make_model=make_mobilenet,
         transform=aug_color_transform, epochs=30, cosine=True),
    # --- the best combinations ---
    dict(name="scratch combo: aug+colour+GAP+dropout+AdamW+LS, 30 ep",
         make_model=lambda: LeafNetReg(gap=True, dropout=0.25), transform=aug_color_transform, epochs=30,
         optimizer="adamw", weight_decay=1e-4, label_smoothing=0.1),
    dict(name="transfer combo + best-epoch weights kept", make_model=make_mobilenet,
         transform=aug_color_transform, epochs=30, cosine=True, weight_decay=1e-4, optimizer="adamw"),
    # --- noise check: repeat the baseline with a second seed ---
    dict(name="B0 baseline, seed 1 (noise check)", make_model=LeafNet, transform=base_transform,
         epochs=15, seed=1),
]

if __name__ == "__main__":
    # epoch lines must reach the terminal (or the log file) as they happen:
    # redirected output is block-buffered by default, which makes a long run look stuck
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]

    if "--list" in args:
        for i, v in enumerate(VARIANTS, 1):
            print(f"{i:2d}  {v['name']}  ({v['epochs']} epochs)")
        raise SystemExit(0)

    # --only 1 2 3 -> just those variant numbers; otherwise the whole list
    start = args.index("--only") + 1 if "--only" in args else None
    picked = {int(a) for a in args[start:] if a.isdigit()} if start else set()
    todo = list(enumerate(VARIANTS, 1))
    if picked:
        todo = [(i, v) for i, v in todo if i in picked]

    rows = [] if "--fresh" in args else load_rows()   # resume an interrupted sweep
    done = {r["name"] for r in rows}
    todo = [(i, v) for i, v in todo if v["name"] not in done]

    print(f"torch {torch.__version__} on {device} — {len(todo)} variant(s) to run, "
          f"{len(done)} already in {RESULTS_CSV.name}")
    for n, (i, variant) in enumerate(todo, 1):
        print(f"\n########## [{n}/{len(todo)}] variant {i} ##########")
        row = run(**variant)
        rows = [r for r in rows if r["name"] != row["name"]] + [row]   # rerunning a variant replaces its row
        rows.sort(key=lambda r: [v["name"] for v in VARIANTS].index(r["name"]))
        save(rows)                       # after every variant, not only at the end
        print(f"saved {len(rows)}/{len(VARIANTS)} rows to {RESULTS_CSV.name} and task4_results.txt")

    print("\n" + "\n".join(summary_table(rows)))
    print(f"summary written to task4_results.txt ({len(rows)} of {len(VARIANTS)} variants)")
