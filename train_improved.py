"""Fine-tuning EfficientNet-B0: augmentasi kuat, class weight, AMP, seleksi by Macro-F1."""
import argparse
import numpy as np, torch, torch.nn as nn, timm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from submission.model import get_model, get_transform, IMG, MEAN, STD


def macro_f1(y, p, k=5):
    f1s = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return sum(f1s) / k


ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--full", action="store_true", help="latih di my_train + my_val (tanpa seleksi)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25),
])
tr = datasets.ImageFolder("my_train", train_tf)
va = datasets.ImageFolder("my_val", get_transform())
print("kelas:", tr.classes, "| train:", len(tr), "| val:", len(va))

if a.full:
    tr_full = datasets.ImageFolder("my_val", train_tf)
    train_ds = ConcatDataset([tr, tr_full])
    targets = tr.targets + tr_full.targets
else:
    train_ds, targets = tr, tr.targets

cnt = np.bincount(targets, minlength=5)
w = torch.tensor(cnt.sum() / (5 * cnt), dtype=torch.float).to(dev)
print("class weight:", w.cpu().numpy().round(2))

dl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=2, drop_last=True)
vl = DataLoader(va, a.bs, num_workers=2)

model = get_model()
pre = timm.create_model("efficientnet_b0", pretrained=True).state_dict()
pre = {k: v for k, v in pre.items() if not k.startswith("classifier.")}
print(model.load_state_dict(pre, strict=False))
model.to(dev)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

best = -1
for ep in range(1, a.epochs + 1):
    model.train()
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()

    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in vl:
            yp += model(x.to(dev)).argmax(1).cpu().tolist(); yt += y.tolist()
    f1 = macro_f1(yt, yp)
    acc = np.mean(np.array(yt) == np.array(yp))
    print(f"epoch {ep:2d}  loss={loss.item():.3f}  val_acc={acc:.4f}  macro_f1={f1:.4f}")
    if a.full:
        torch.save(model.state_dict(), "submission/model.pth")
    elif f1 > best:
        best = f1
        torch.save(model.state_dict(), "submission/model.pth")
print("selesai ->", "epoch terakhir" if a.full else f"best macro_f1={best:.4f}", "disimpan di submission/model.pth")
