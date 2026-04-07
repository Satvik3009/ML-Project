import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR

import pickle
import nibabel as nib
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModel

# ── Paths (all relative — no hardcoded absolute paths) ───────────────────────
from paths import P

P.makedirs()

# FER2013 folder  (lowercase 'fer2013', matching the actual folder on disk)
FER_PATH  = P.FER_TRAIN.parent          # …/data/fer2013
DEAP_PATH = P.DEAP_DAT                  # …/data/deap-dataset/data_preprocessed_python
FMRI_PATH = P.FMRI                      # …/data/fmri

MODEL_PATH = r"C:\Users\HP\ML_project\emotion_model_v4.pth"

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    batch_size        = 32,         # safe for T4 with frozen ViT
    epochs            = 15,
    lr                = 3e-4,
    weight_decay      = 1e-4,
    dropout           = 0.3,
    grad_clip         = 1.0,
    warmup_epochs     = 2,
    eeg_dim           = 32 * 5,
    fmri_dim          = 32,
    face_dim          = 768,        # ViT-B/16 [CLS] hidden size
    num_classes       = 7,
    img_size          = 224,        # ViT-B/16 native resolution
    freeze_vit_layers = 12,         # all 12 ViT-B/16 blocks frozen
)

# ── FER2013 transforms ────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ImageFolder expects:  data/fer2013/train/angry/*.jpg  etc.
train_data = ImageFolder(str(P.FER_TRAIN), transform=train_transform)
test_data  = ImageFolder(str(P.FER_TEST),  transform=test_transform)

# Weighted sampler — fixes class imbalance
class_counts   = np.bincount([s[1] for s in train_data.samples])
class_weights  = 1.0 / class_counts
sample_weights = torch.tensor(
    [class_weights[s[1]] for s in train_data.samples], dtype=torch.float
)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
print("Class counts:", dict(zip(train_data.classes, class_counts)))

train_loader = DataLoader(train_data, batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=CFG["batch_size"],
                          num_workers=0, pin_memory=True)

emotion_labels = train_data.classes
print("Emotion classes:", emotion_labels)

# ── Face Encoder — ViT fully frozen ──────────────────────────────────────────
class FaceEncoder(nn.Module):
    def __init__(self,
                 model_name="trpakov/vit-face-expression",
                 dropout=0.3,
                 freeze_layers=12):      # kept for API compatibility
        super().__init__()
        self.vit  = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.drop = nn.Dropout(p=dropout)

        # Freeze every parameter in the ViT backbone (embeddings + all 12 blocks + pooler)
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):                    # x: (B, 1, 224, 224) grayscale
        x   = x.repeat(1, 3, 1, 1)          # → (B, 3, 224, 224)
        with torch.no_grad():                # no grad computation through backbone
            out = self.vit(pixel_values=x)
        cls = out.last_hidden_state[:, 0]    # [CLS] token → (B, 768)
        return self.drop(cls)

# ── EEG feature extraction ────────────────────────────────────────────────────
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

def bandpower(psd, freqs, low, high):
    idx = np.where((freqs >= low) & (freqs < high))[0]
    return np.mean(psd[idx]) if len(idx) else 0.0

def extract_eeg_features():
    features, labels = [], []
    if not DEAP_PATH.exists():
        print(f"  [EEG] DEAP path not found: {DEAP_PATH} — skipping")
        return np.zeros((1, CFG["eeg_dim"]), dtype=np.float32), np.zeros(1, dtype=np.int64)

    for file in os.listdir(DEAP_PATH):
        path = DEAP_PATH / file
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, :]
        lab = data["labels"]
        for trial in range(eeg.shape[0]):
            psd_features = []
            for ch in range(32):
                freqs, psd = welch(eeg[trial, ch], fs=128, nperseg=256)
                for low, high in BANDS.values():
                    psd_features.append(bandpower(psd, freqs, low, high))
            features.append(psd_features)
            labels.append(lab[trial][0])
    X     = np.array(features, dtype=np.float32)
    y     = np.array(labels,   dtype=np.float32)
    y_bin = (y > 5).astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y_bin

print("Extracting EEG features...")
eeg_X, eeg_y = extract_eeg_features()
print(f"  EEG: {eeg_X.shape}")

# ── fMRI feature extraction ───────────────────────────────────────────────────
def extract_fmri_features(target_dim=32):
    features = []
    if not FMRI_PATH.exists():
        print(f"  [fMRI] Path not found: {FMRI_PATH} — fMRI branch disabled")
        return None

    for sub in os.listdir(FMRI_PATH):
        sub_path = FMRI_PATH / sub
        if not sub_path.is_dir():
            continue
        for file in os.listdir(sub_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                img  = nib.load(sub_path / file)
                data = img.get_fdata()
                if data.ndim == 4:
                    vec = data.reshape(-1, data.shape[-1]).mean(axis=1)
                else:
                    vec = data.flatten()
                vec = vec[:target_dim] if len(vec) >= target_dim else \
                      np.pad(vec, (0, target_dim - len(vec)))
                features.append(vec.astype(np.float32))
    if not features:
        print("  No fMRI files found — fMRI branch disabled.")
        return None
    X = np.array(features)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X

print("Extracting fMRI features...")
fmri_X = extract_fmri_features(target_dim=CFG["fmri_dim"])
if fmri_X is not None:
    print(f"  fMRI: {fmri_X.shape}")

# ── EEG Encoder ──────────────────────────────────────────────────────────────
class EEGEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

# ── Fusion Classifier ─────────────────────────────────────────────────────────
class FusionClassifier(nn.Module):
    def __init__(self, use_fmri=False, fmri_dim=32, num_classes=7, dropout=0.3):
        super().__init__()
        self.face_enc = FaceEncoder(dropout=dropout,
                                    freeze_layers=CFG["freeze_vit_layers"])
        self.eeg_enc  = EEGEncoder(CFG["eeg_dim"], dropout=dropout)
        self.use_fmri = use_fmri
        fused_dim     = 768 + 64
        if use_fmri:
            self.fmri_enc = nn.Sequential(nn.Linear(fmri_dim, 32), nn.ReLU())
            fused_dim += 32
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, eeg, fmri=None):
        parts = [self.face_enc(img), self.eeg_enc(eeg)]
        if self.use_fmri and fmri is not None:
            parts.append(self.fmri_enc(fmri))
        return self.classifier(torch.cat(parts, dim=1))


use_fmri = fmri_X is not None
model    = FusionClassifier(
    use_fmri    = use_fmri,
    fmri_dim    = CFG["fmri_dim"],
    num_classes = CFG["num_classes"],
    dropout     = CFG["dropout"],
).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\nModel ready | use_fmri={use_fmri}")
print(f"Trainable: {trainable:,}  Frozen: {frozen:,}")

# ── Load checkpoint if available ─────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f"\nFound checkpoint — loading (strict=False)...")
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print("  Loaded.")
    except Exception as e:
        print(f"  Could not load checkpoint ({e}). Training from pretrained ViT.")
else:
    print("\nNo checkpoint — training from pretrained ViT backbone.")

# ── Optimizer — head only (backbone fully frozen, no backbone param group) ───
head_params = (
    list(model.face_enc.drop.parameters()) +
    list(model.eeg_enc.parameters()) +
    list(model.classifier.parameters())
)
if use_fmri:
    head_params += list(model.fmri_enc.parameters())

optimizer = torch.optim.AdamW(
    head_params,
    lr           = CFG["lr"],
    weight_decay = CFG["weight_decay"],
)

# ── Scheduler — linear warmup + cosine decay ─────────────────────────────────
steps_per_epoch = len(train_loader)
total_steps     = CFG["epochs"] * steps_per_epoch
warmup_steps    = CFG["warmup_epochs"] * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)

# ── Loss ─────────────────────────────────────────────────────────────────────
per_class_weights = torch.tensor([1.5, 2.0, 1.5, 0.8, 1.0, 1.5, 0.8], device=device)
criterion = nn.CrossEntropyLoss(weight=per_class_weights, label_smoothing=0.1)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n--- Training on FER2013 (frozen ViT backbone) ---")
best_acc, best_state = 0.0, None

for epoch in range(CFG["epochs"]):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        eeg_dummy  = torch.zeros(imgs.size(0), CFG["eeg_dim"],  device=device)
        fmri_dummy = (torch.zeros(imgs.size(0), CFG["fmri_dim"], device=device)
                      if use_fmri else None)

        optimizer.zero_grad()
        logits = model(imgs, eeg_dummy, fmri_dummy)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = 100.0 * correct / total
    cur_lr    = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{CFG['epochs']}  "
          f"Loss: {total_loss:.3f}  "
          f"Train Acc: {train_acc:.1f}%  "
          f"LR: {cur_lr:.2e}")

    if train_acc > best_acc:
        best_acc   = train_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

torch.save(best_state, MODEL_PATH)
print(f"\nBest train acc: {best_acc:.1f}%  |  Model saved → {MODEL_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs       = imgs.to(device)
        eeg_dummy  = torch.zeros(imgs.size(0), CFG["eeg_dim"],  device=device)
        fmri_dummy = (torch.zeros(imgs.size(0), CFG["fmri_dim"], device=device)
                      if use_fmri else None)
        logits = model(imgs, eeg_dummy, fmri_dummy)
        y_true.extend(labels.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=emotion_labels, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Emotion Recognition — Confusion Matrix v4 (Frozen ViT)", fontsize=16)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
out_path = str(P.OUTPUTS / "confusion_matrix_v4.png")
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Confusion matrix saved → {out_path}")