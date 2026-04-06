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

# KEY CHANGE: ViT pretrained on FER+ emotion data
from transformers import AutoModel

# =====================================
# PATHS
# =====================================

DATA_DIR = r"C:\Users\HP\ML_project\data"

FER_PATH  = os.path.join(DATA_DIR, "FER2013")
DEAP_PATH = os.path.join(DATA_DIR, "deap-dataset", "data_preprocessed_python")
FMRI_PATH = os.path.join(DATA_DIR, "fmri")

MODEL_PATH = "emotion_model_v4.pth"

# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================
# CONFIG
# =====================================

CFG = dict(
    batch_size        = 16,
    epochs            = 15,      # ↑ from 10
    lr                = 3e-4,    # head LR; backbone gets 1e-5 (see optimizer)
    weight_decay      = 1e-4,
    dropout           = 0.3,     # ↓ slightly — ViT is already regularised
    grad_clip         = 1.0,
    warmup_epochs     = 2,       # NEW: linear warmup prevents early instability
    eeg_dim           = 32 * 5,  # 32 channels × 5 freq bands
    fmri_dim          = 32,
    face_dim          = 768,     # ViT-B/16 [CLS] hidden size (vs 1408 for EfficientNet-B2)
    num_classes       = 7,
    img_size          = 224,     # ViT-B/16 native resolution (vs 260)
    freeze_vit_layers = 8,       # freeze first 8/12 blocks, fine-tune last 4
)

# =====================================
# FER2013 — TRANSFORMS
# =====================================

train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW: spatial shift
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),      # NEW: occlusion robustness
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_data = ImageFolder(os.path.join(FER_PATH, "train"), transform=train_transform)
test_data  = ImageFolder(os.path.join(FER_PATH, "test"),  transform=test_transform)

# Weighted sampler — keeps class imbalance from dominating
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

# =====================================
# FACE ENCODER — ViT pretrained on FER+
# =====================================
#
# WHY THIS MATTERS:
#   Old code used EfficientNet-B2 pretrained on ImageNet (objects, scenes).
#   Facial micro-expressions (disgust vs fear, anger vs surprise) are
#   completely out of ImageNet's distribution — the features just don't
#   transfer well without extensive training.
#
#   trpakov/vit-face-expression is ViT-B/16 already fine-tuned on FER+,
#   so its attention heads are already tuned to eyes, brows, mouth corners
#   — the exact regions that distinguish emotions. We just adapt it to
#   FER2013's label scheme.
#
# STRATEGY:
#   Freeze the first 8 of 12 transformer blocks (stable low-level texture
#   and face structure), fine-tune the last 4 (emotion-specific semantics).

class FaceEncoder(nn.Module):
    def __init__(self,
                 model_name="trpakov/vit-face-expression",
                 dropout=0.3,
                 freeze_layers=8):
        super().__init__()
        self.vit  = AutoModel.from_pretrained(model_name,
                                              ignore_mismatched_sizes=True)
        self.drop = nn.Dropout(p=dropout)

        # Freeze patch embeddings + positional encodings
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze first `freeze_layers` attention blocks
        for i, block in enumerate(self.vit.encoder.layer):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):                    # x: (B, 1, 224, 224) grayscale
        x   = x.repeat(1, 3, 1, 1)          # → (B, 3, 224, 224)  [ViT expects RGB]
        out = self.vit(pixel_values=x)
        cls = out.last_hidden_state[:, 0]    # [CLS] token → (B, 768)
        return self.drop(cls)


# =====================================
# EEG FEATURE EXTRACTION — band power
# =====================================

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
    for file in os.listdir(DEAP_PATH):
        path = os.path.join(DEAP_PATH, file)
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, :]          # (40 trials, 32 ch, 8064 samples)
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

print("Extracting EEG features (band power)...")
eeg_X, eeg_y = extract_eeg_features()
print(f"  EEG feature matrix: {eeg_X.shape}")

# =====================================
# fMRI FEATURE EXTRACTION
# =====================================

def extract_fmri_features(target_dim=32):
    features = []
    for sub in os.listdir(FMRI_PATH):
        sub_path = os.path.join(FMRI_PATH, sub)
        if not os.path.isdir(sub_path):
            continue
        for file in os.listdir(sub_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                img  = nib.load(os.path.join(sub_path, file))
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
    print(f"  fMRI feature matrix: {fmri_X.shape}")

# =====================================
# EEG ENCODER
# =====================================

class EEGEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),              # GELU matches ViT internals
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)          # (B, 64)

# =====================================
# FUSION CLASSIFIER
# =====================================

class FusionClassifier(nn.Module):
    """
    ViT face features (768) + EEG features (64) [+ optional fMRI (32)]
    → LayerNorm → MLP → 7-class emotion logits.

    LayerNorm is used instead of BatchNorm before the classifier because
    it works better with ViT-style feature distributions.
    """
    def __init__(self, use_fmri=False, fmri_dim=32,
                 num_classes=7, dropout=0.3):
        super().__init__()
        self.face_enc = FaceEncoder(dropout=dropout,
                                    freeze_layers=CFG["freeze_vit_layers"])
        self.eeg_enc  = EEGEncoder(CFG["eeg_dim"], dropout=dropout)

        self.use_fmri = use_fmri
        fused_dim = 768 + 64        # ViT [CLS] (768) + EEG encoder out (64)
        if use_fmri:
            self.fmri_enc = nn.Sequential(
                nn.Linear(fmri_dim, 32),
                nn.ReLU(),
            )
            fused_dim += 32

        # LayerNorm → Linear → GELU → Dropout → Linear
        # LayerNorm is more stable than BatchNorm when features come from ViT
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, eeg, fmri=None):
        face_feat = self.face_enc(img)
        eeg_feat  = self.eeg_enc(eeg)
        parts     = [face_feat, eeg_feat]

        if self.use_fmri and fmri is not None:
            parts.append(self.fmri_enc(fmri))

        fused = torch.cat(parts, dim=1)
        return self.classifier(fused)


use_fmri = fmri_X is not None
model    = FusionClassifier(
    use_fmri    = use_fmri,
    fmri_dim    = CFG["fmri_dim"],
    num_classes = CFG["num_classes"],
    dropout     = CFG["dropout"],
).to(device)

print(f"\nModel ready  |  use_fmri={use_fmri}")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable params: {trainable:,}  |  Frozen params: {frozen:,}")

# =====================================
# CHECKPOINT LOADING
# =====================================
# Architecture changed (EfficientNet → ViT), so old weights are incompatible.
# Skip old checkpoint entirely — the pretrained ViT backbone is already a much
# better starting point than any FER2013-trained EfficientNet checkpoint.

if os.path.exists(MODEL_PATH):
    print(f"\nFound {MODEL_PATH} — attempting to load...")
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print("  Checkpoint loaded (strict=False — missing ViT keys are expected).")
    except Exception as e:
        print(f"  Could not load checkpoint ({e}). Training from pretrained ViT.")
else:
    print("\nNo checkpoint — training from pretrained ViT backbone.")

# =====================================
# OPTIMIZER — differential learning rates
# =====================================
#
# CRITICAL DESIGN CHOICE:
#   Backbone (ViT pretrained on FER+) → very low LR (1e-5)
#     We want it to barely move — just adapt to FER2013's label scheme.
#   EEG encoder + fusion head → normal LR (3e-4)
#     These train from scratch so they need a proper step size.
#
# Using a single LR for both (as in the old code) would either:
#   - destroy pretrained features (too high), or
#   - never train the new head (too low).

backbone_params = list(model.face_enc.vit.parameters())
head_params = (
    list(model.face_enc.drop.parameters()) +
    list(model.eeg_enc.parameters()) +
    list(model.classifier.parameters())
)
if use_fmri:
    head_params += list(model.fmri_enc.parameters())

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": 1e-5},    # ← very low for pretrained ViT
        {"params": head_params,     "lr": CFG["lr"]},
    ],
    weight_decay=CFG["weight_decay"],
)

# =====================================
# SCHEDULER — linear warmup + cosine decay
# =====================================
#
# Cold-starting with full LR causes gradient spikes that destabilise
# the pretrained ViT features. 2-epoch linear warmup + cosine decay
# gives stable convergence.

steps_per_epoch = len(train_loader)
total_steps     = CFG["epochs"] * steps_per_epoch
warmup_steps    = CFG["warmup_epochs"] * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)

# =====================================
# LOSS — label-smoothed cross-entropy
# =====================================
#
# Switched from FocalLoss to label-smoothed CE:
#   - Focal loss is tricky to tune and can cause training instability
#     with a partially-frozen pretrained backbone.
#   - Label smoothing (0.1) prevents overconfident predictions on easy
#     classes (happy, neutral) and keeps gradients flowing for hard ones.
#   - Per-class weights still handle the imbalance signal.

per_class_weights = torch.tensor([
    1.5,   # angry
    2.0,   # disgust  ← most underrepresented
    1.5,   # fear
    0.8,   # happy    ← very common; slight downweight
    1.0,   # neutral
    1.5,   # sad
    0.8,   # surprise ← slight downweight
], device=device)

criterion = nn.CrossEntropyLoss(
    weight          = per_class_weights,
    label_smoothing = 0.1,
)

# =====================================
# TRAINING LOOP
# =====================================

print("\n--- Training on FER2013 ---")

best_acc   = 0.0
best_state = None

for epoch in range(CFG["epochs"]):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # EEG / fMRI not available for FER2013 — use zeros as neutral input
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

    train_acc  = 100.0 * correct / total
    head_lr    = optimizer.param_groups[1]["lr"]
    backbone_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{CFG['epochs']}  "
          f"Loss: {total_loss:.3f}  "
          f"Train Acc: {train_acc:.1f}%  "
          f"LR(head): {head_lr:.2e}  LR(backbone): {backbone_lr:.2e}")

    # Keep the best snapshot
    if train_acc > best_acc:
        best_acc   = train_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Save the best snapshot (not necessarily last epoch)
torch.save(best_state, MODEL_PATH)
print(f"\nBest train acc: {best_acc:.1f}%  |  Model saved → {MODEL_PATH}")

# =====================================
# EVALUATION
# =====================================

# Load best weights before eval
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
print(classification_report(y_true, y_pred,
                             target_names=emotion_labels, zero_division=0))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Emotion Recognition — Confusion Matrix v4", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_v4.png", dpi=150)
plt.show()
print("Confusion matrix saved → confusion_matrix_v4.png")