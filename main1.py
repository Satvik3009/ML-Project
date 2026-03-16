"""
=============================================================================
FULL ML PIPELINE: Multi-Modal Emotion Recognition → Brain Tumor Detection
=============================================================================

REAL DATASET FORMATS (what you actually get after Kaggle download):

  1. FER2013 (msambare/fer2013)
     ├── train/
     │   ├── angry/    *.jpg
     │   ├── disgust/  *.jpg
     │   ├── fear/     *.jpg
     │   ├── happy/    *.jpg
     │   ├── neutral/  *.jpg
     │   ├── sad/      *.jpg
     │   └── surprise/ *.jpg
     └── test/
         └── {same 7 emotion folders}/

  2. fMRI dataset (irajahangari/fmri-dataset-for-emotion-recognition)
     NIfTI volumetric brain scans — NOT regular images.
     ├── sub-01/
     │   ├── func/sub-01_task-emotion_bold.nii.gz  (4D volume: X×Y×Z×time)
     │   └── labels.csv  (columns: TR, emotion)
     └── sub-02/ ...

  3. DEAP EEG
     ├── s01.dat  (Python pickle: {data: (40,40,8064), labels: (40,4)})
     └── ... s32.dat

  4. MNE Sample EEG (mubin986/mne-sample-data-processed)
     ├── epochs_data.npy    (N, channels, timepoints)
     └── epochs_labels.npy  (N,)

  5. FigShare Brain Tumor (ashkhagan/figshare-brain-tumor-dataset)
     ├── no_tumor/    *.jpg
     ├── glioma/      *.jpg
     ├── meningioma/  *.jpg
     └── pituitary/   *.jpg

INSTALL:
  pip install torch torchvision nibabel scipy scikit-learn pandas numpy pillow opencv-python
=============================================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder

import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "fer_train_dir"  : "data/fer/train",
    "fer_test_dir"   : "data/fer/test",
    "fmri_root"      : "data/fmri",
    "deap_dir"       : "data/deap-dataset",
    "mne_dir"        : "data/MNE-sample-data",
    "tumor_dir"      : "data/dataset/data",
    "batch_size"     : 32,
    "epochs_emotion" : 20,
    "epochs_tumor"   : 15,
    "lr"             : 1e-3,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    "n_fer_classes"  : 7,
    "n_tumor_classes": 4,
    "tumor_classes"  : ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
}
DEVICE = torch.device(CFG["device"])
print(f"[Config] Device: {DEVICE}")


# =============================================================================
# DATASET 1 — FER2013  (JPEG images in emotion subfolders)
# =============================================================================
# The Kaggle FER2013 dataset by msambare contains JPEG images arranged in
# subfolders named by emotion class.  PyTorch ImageFolder handles this
# pattern automatically — it walks the directory, assigns integer labels
# by sorted folder name, and returns (image_tensor, label_int).
#
# Images are 48×48 grayscale. We replicate them to 3 channels so that
# ImageNet-pretrained ResNet weights can be applied directly.
# =============================================================================

def build_fer_loaders(train_dir, test_dir, batch_size):
    """
    Returns: train_loader, val_loader, test_loader, class_to_idx

    Folder structure:
        train_dir/angry/*.jpg
        train_dir/happy/*.jpg  ... etc.
    """
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),   # grayscale → 3ch for ResNet
        T.Resize((48, 48)),
        T.RandomHorizontalFlip(p=0.5),        # faces are left-right symmetric
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    eval_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    train_ds  = ImageFolder(root=train_dir, transform=train_tf)
    full_test = ImageFolder(root=test_dir,  transform=eval_tf)

    # Split test set into 50% val / 50% held-out test
    n_val = len(full_test) // 2
    val_ds, test_ds = random_split(full_test, [n_val, len(full_test)-n_val],
                                   generator=torch.Generator().manual_seed(42))

    print(f"[FER2013] train={len(train_ds):,}  val={n_val:,}  test={len(full_test)-n_val:,}")
    print(f"[FER2013] labels: {train_ds.class_to_idx}")

    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(val_ds,   batch_size, shuffle=False, num_workers=4),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=4),
            train_ds.class_to_idx)


# =============================================================================
# DATASET 2 — fMRI  (NIfTI 4D volumetric brain scans)
# =============================================================================
# fMRI data is NOT regular images.  Each file is a 4-dimensional NIfTI volume:
#
#   shape = (X, Y, Z, T)   e.g. (64, 64, 36, 200)
#   X, Y, Z  = spatial voxel grid of the brain
#   T        = time points (brain states sampled every ~2 s)
#
# A companion labels.csv maps each TR (time point index) to an emotion label.
# We extract individual 3D volumes (one per labeled TR) and feed each volume
# through a 3D CNN that learns spatial patterns across brain regions.
#
# Requires: pip install nibabel
# =============================================================================

class FMRIEmotionDataset(Dataset):
    """
    Loads 4D NIfTI fMRI volumes and pairs each labeled time-point with
    its emotion label.

    Expected layout under fmri_root:
        sub-01/
          func/sub-01_task-emotion_bold.nii.gz
          labels.csv   (columns: TR, emotion)
        sub-02/ ...

    Each __getitem__ returns:
        vol:   (1, X, Y, Z) float32 tensor — single 3D brain volume
        label: int tensor
    """
    def __init__(self, fmri_root, target_shape=(32, 32, 20)):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel:  pip install nibabel")

        self.volumes  = []
        self.labels   = []
        self.target   = target_shape
        raw_labels    = []

        for subj in sorted(Path(fmri_root).iterdir()):
            if not subj.is_dir():
                continue
            niis   = list(subj.glob("func/*.nii.gz")) + list(subj.glob("func/*.nii")) \
                   + list(subj.glob("*.nii.gz"))      + list(subj.glob("*.nii"))
            lbls   = list(subj.glob("labels.csv"))    + list(subj.glob("func/labels.csv"))
            if not niis or not lbls:
                print(f"  [fMRI] skipping {subj.name} — no .nii or labels.csv")
                continue

            data4d = nib.load(str(niis[0])).get_fdata()   # (X, Y, Z, T)
            df     = pd.read_csv(lbls[0])                  # columns: TR, emotion

            for _, row in df.iterrows():
                tr = int(row['TR'])
                if tr >= data4d.shape[3]:
                    continue
                vol = self._process(data4d[:, :, :, tr])
                self.volumes.append(vol)
                raw_labels.append(str(row['emotion']).strip().lower())

        if not self.volumes:
            raise ValueError(f"No fMRI samples loaded from {fmri_root}. "
                             "Check folder structure and labels.csv format.")

        le               = LabelEncoder()
        self.labels      = le.fit_transform(raw_labels)
        self.n_classes   = len(le.classes_)
        self.class_names = list(le.classes_)
        print(f"[fMRI] {len(self.volumes):,} volumes | classes={self.class_names}")

    def _process(self, vol):
        """Resize 3D volume to target_shape and z-score normalise."""
        from scipy.ndimage import zoom
        factors = [t/s for t,s in zip(self.target, vol.shape)]
        v = zoom(vol, factors, order=1).astype(np.float32)
        return (v - v.mean()) / (v.std() + 1e-8)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.volumes[i]).unsqueeze(0),   # (1,X,Y,Z)
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def build_fmri_loaders(fmri_root, batch_size):
    ds = FMRIEmotionDataset(fmri_root)
    n_val = int(0.2 * len(ds))
    tr_ds, v_ds = random_split(ds, [len(ds)-n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr_ds, batch_size, shuffle=True,  num_workers=2),
            DataLoader(v_ds,  batch_size, shuffle=False, num_workers=2),
            ds.n_classes)


# =============================================================================
# DATASET 3 — EEG  (DEAP .dat pickles + MNE numpy arrays)
# =============================================================================
# DEAP format — each .dat file is a Python2 pickle:
#   'data'  : ndarray (40 trials, 40 channels, 8064 timepoints @ 128 Hz)
#              channels 0-31 = EEG;  32-39 = peripheral physiology (skip)
#   'labels': ndarray (40 trials, 4) = valence, arousal, dominance, liking
#
# Feature extraction per trial:
#   For each of 5 EEG frequency bands (δ θ α β γ):
#     1. Bandpass filter with 4th-order Butterworth
#     2. Welch PSD — average power per channel in this band
#     3. Differential Entropy (DE) = 0.5 * log(2πe * variance)
#   Final feature vector: 32 channels × 5 bands × 2 features = 320-dim
#
# MNE fallback: load pre-exported epochs_data.npy + epochs_labels.npy
# =============================================================================

class EEGEmotionDataset(Dataset):
    BANDS = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,50)}

    def __init__(self, deap_dir=None, mne_dir=None, fs=128):
        self.fs, self.features, self.labels = fs, [], []
        deap_files = list(Path(deap_dir).glob("s*.dat")) if deap_dir else []
        if deap_files:
            print(f"[EEG] DEAP: {len(deap_files)} subjects")
            self._load_deap(deap_dir)
        elif mne_dir and (Path(mne_dir)/"epochs_data.npy").exists():
            print(f"[EEG] MNE arrays from {mne_dir}")
            self._load_mne(mne_dir)
        else:
            raise FileNotFoundError("Provide deap_dir (s*.dat files) or mne_dir (.npy files)")

        self.features = np.stack(self.features).astype(np.float32)
        self.labels   = np.array(self.labels, dtype=np.int64)
        print(f"[EEG] features={self.features.shape}  n_classes={len(np.unique(self.labels))}")

    def _extract(self, trial):
        """
        trial: (n_channels, n_timepoints)
        Returns: flat feature vector  (n_channels * n_bands * 2,)

        PSD  — how much oscillation power exists in this frequency band
        DE   — differential entropy: log-variance measure, more stable than raw PSD
        """
        out = []
        for lo, hi in self.BANDS.values():
            nyq  = self.fs / 2.0
            b, a = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype='band')
            filt = scipy.signal.filtfilt(b, a, trial, axis=-1)   # zero-phase filter
            _, psd = scipy.signal.welch(filt, fs=self.fs, nperseg=256, axis=-1)
            psd_mean = psd.mean(axis=-1)                          # (C,)
            de = 0.5 * np.log(2*np.pi*np.e * np.var(filt, axis=-1) + 1e-8)  # (C,)
            out.extend([psd_mean, de])
        return np.concatenate(out)

    def _load_deap(self, deap_dir):
        for path in sorted(Path(deap_dir).glob("s*.dat")):
            with open(path,'rb') as f:
                s = pickle.load(f, encoding='latin1')
            eeg    = s['data'][:, :32, 3*self.fs:]   # skip 3s baseline, EEG channels only
            labels = s['labels']
            for t in range(eeg.shape[0]):
                self.features.append(self._extract(eeg[t]))
                v, a = labels[t,0], labels[t,1]
                self.labels.append(1 if (v>=5 and a>=5) else 0)  # binarise valence×arousal

    def _load_mne(self, mne_dir):
        data   = np.load(Path(mne_dir)/"epochs_data.npy")
        labels = np.load(Path(mne_dir)/"epochs_labels.npy")
        for i in range(len(data)):
            self.features.append(self._extract(data[i]))
            self.labels.append(int(labels[i]))

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.features[i]), torch.tensor(int(self.labels[i])))


def build_eeg_loaders(deap_dir, mne_dir, batch_size):
    ds = EEGEmotionDataset(deap_dir=deap_dir, mne_dir=mne_dir)
    feat_dim  = ds.features.shape[1]
    n_classes = len(np.unique(ds.labels))
    tr_i, v_i = train_test_split(range(len(ds)), test_size=0.2,
                                  stratify=ds.labels, random_state=42)
    return (DataLoader(Subset(ds,tr_i), batch_size, shuffle=True),
            DataLoader(Subset(ds,v_i),  batch_size, shuffle=False),
            feat_dim, n_classes)


# =============================================================================
# DATASET 4 — Brain Tumor MRI  (JPEG images in class subfolders)
# =============================================================================
# FigShare Brain Tumor has the same ImageFolder pattern as FER2013.
# We use a custom Dataset (not ImageFolder) because each sample also needs
# to carry an emotion embedding vector for the conditioned classifier.
# =============================================================================

class BrainTumorDataset(Dataset):
    """
    Loads brain MRI JPEGs from class-labelled subfolders.
    Optionally attaches a pre-computed emotion probability vector to each sample.

    __getitem__ returns: (image_tensor, tumor_label, emotion_vec)
    """
    def __init__(self, root_dir, transform=None, emotion_vectors=None, n_emotions=7):
        self.paths          = []
        self.labels         = []
        self.transform      = transform
        self.emotion_vectors = emotion_vectors or {}
        self.n_emotions     = n_emotions
        raw_labels          = []

        for cls_dir in sorted(Path(root_dir).iterdir()):
            if not cls_dir.is_dir(): continue
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.paths.append(str(img_path))
                raw_labels.append(cls_dir.name.lower())

        le               = LabelEncoder()
        self.labels      = le.fit_transform(raw_labels)
        self.class_names = list(le.classes_)
        print(f"[Tumor] {len(self.paths):,} images | classes={self.class_names}")

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        fname = os.path.basename(self.paths[i])
        e_vec = self.emotion_vectors.get(fname, np.zeros(self.n_emotions, dtype=np.float32))
        return img, int(self.labels[i]), torch.tensor(e_vec, dtype=torch.float32)


def build_tumor_loaders(tumor_dir, batch_size, emotion_vectors=None, n_emotions=7):
    train_tf = T.Compose([
        T.Resize((128,128)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(15), T.ColorJitter(brightness=0.15, contrast=0.2),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((128,128)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    full   = BrainTumorDataset(tumor_dir, train_tf, emotion_vectors, n_emotions)
    val_ds = BrainTumorDataset(tumor_dir, eval_tf,  emotion_vectors, n_emotions)
    tr_i, v_i = train_test_split(range(len(full)), test_size=0.2,
                                  stratify=full.labels, random_state=42)
    return (DataLoader(Subset(full,   tr_i), batch_size, shuffle=True,  num_workers=4),
            DataLoader(Subset(val_ds, v_i),  batch_size, shuffle=False, num_workers=4),
            full)


# =============================================================================
# MODELS
# =============================================================================

class EmotionCNN2D(nn.Module):
    """
    ResNet18 fine-tuned for 2D emotion classification (FER2013 faces).

    Skip connections in ResNet prevent vanishing gradients.
    ImageNet pre-training provides strong low-level feature detectors.
    A forward hook captures 512-dim pooled features for fusion.
    """
    def __init__(self, n_classes, dropout=0.4):
        super().__init__()
        base = tv_models.resnet18(weights="IMAGENET1K_V1")
        self.feature_dim = base.fc.in_features   # 512
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )
        self.net  = base
        self._emb = None
        self.net.avgpool.register_forward_hook(
            lambda m,i,o: setattr(self, '_emb', o.flatten(1)))

    def forward(self, x):             return self.net(x)
    def get_embedding(self, x):       self.forward(x); return self._emb


class fMRIEmotionCNN3D(nn.Module):
    """
    Lightweight 3D CNN for volumetric NIfTI fMRI brain scans.

    Uses Conv3d because fMRI data has 3 spatial dimensions (X, Y, Z).
    Pre-trained 2D CNNs cannot be applied to 3D volumetric data.
    Three conv blocks progressively abstract from voxels → regions → networks.
    AdaptiveAvgPool3d ensures a fixed output size regardless of input volume shape.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16),  nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32, 3, padding=1), nn.BatchNorm3d(32),  nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64, 3, padding=1), nn.BatchNorm3d(64),  nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)),  # → (B, 64, 2, 2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64*2*2*2, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):         return self.classifier(self.encoder(x))
    def get_embedding(self, x):   return self.encoder(x).flatten(1)


class EEGEmotionMLP(nn.Module):
    """
    3-layer MLP for pre-extracted EEG band-power features.

    ELU activations: negative values for x<0 keep activations zero-centred,
    speeding convergence compared to ReLU for this type of feature data.
    BatchNorm stabilises training when feature scales differ across bands.
    A hook captures 64-dim embeddings for fusion.
    """
    def __init__(self, n_classes, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim,256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256,128),      nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.4),
            nn.Linear(128,64),       nn.ELU(),
            nn.Linear(64, n_classes),
        )
        self._emb = None
        self.net[8].register_forward_hook(lambda m,i,o: setattr(self,'_emb',o))

    def forward(self, x):         return self.net(x)
    def get_embedding(self, x):   self.forward(x); return self._emb


class EmotionFusionLayer(nn.Module):
    """
    Late fusion: weighted average of softmax outputs from all 3 emotion models.

    Learnable scalar weights (softmax-normalised) let the model learn
    which modality to trust more for a given emotion.
    Late fusion is preferred over early fusion here because each modality
    (face, fMRI, EEG) has different noise characteristics and temporal resolution.
    """
    def __init__(self, n_modalities=3, n_classes=7):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)
        self.refine  = nn.Sequential(nn.Linear(n_classes,32), nn.ReLU(), nn.Linear(32,n_classes))

    def forward(self, *logits_list):
        w     = F.softmax(self.weights, dim=0)
        probs = [F.softmax(lg, dim=-1) for lg in logits_list]
        fused = sum(w[i]*p for i,p in enumerate(probs))
        return F.softmax(self.refine(fused), dim=-1)


class EmotionConditionedTumorCNN(nn.Module):
    """
    Brain tumor classifier that is conditioned on the predicted emotion state.

    Architecture:
      ResNet50 backbone → 2048-dim MRI features
      Small MLP         → 64-dim emotion features
      Concatenate       → 2112-dim fused vector
      Classifier head   → 4 tumor classes

    Why condition on emotion?
      Emotional state (chronic fear, depression) affects neurological conditions.
      Stress hormones alter brain tissue and blood flow patterns visible in MRI.
      By adding emotion context, the classifier can exploit this auxiliary signal.
    """
    def __init__(self, n_tumor_classes=4, emotion_dim=7):
        super().__init__()
        base = tv_models.resnet50(weights="IMAGENET1K_V1")
        self.mri_encoder     = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.emotion_encoder = nn.Sequential(nn.Linear(emotion_dim,64), nn.ReLU(), nn.Linear(64,64))
        self.classifier      = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048+64, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_tumor_classes),
        )

    def forward(self, img, emotion_vec):
        mri  = self.mri_encoder(img).flatten(1)   # (B, 2048)
        emo  = self.emotion_encoder(emotion_vec)   # (B, 64)
        return self.classifier(torch.cat([mri, emo], dim=1))


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, is_tumor=False):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        if is_tumor:
            x, y, e = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits   = model(x, e)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()*y.size(0); correct += (logits.argmax(1)==y).sum().item(); n += y.size(0)
    return total_loss/n, correct/n


@torch.no_grad()
def eval_epoch(model, loader, device, is_tumor=False):
    model.eval(); yt, yp = [], []
    for batch in loader:
        if is_tumor:
            x, y, e = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits   = model(x, e)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
        yt.extend(y.cpu().numpy()); yp.extend(logits.argmax(1).cpu().numpy())
    yt, yp = np.array(yt), np.array(yp)
    return (yt==yp).mean(), yt, yp


def train_model(model, tr_l, val_l, n_epochs, lr, device, is_tumor=False, name="model"):
    os.makedirs("checkpoints", exist_ok=True)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, n_epochs+1):
        loss, tr_acc = train_epoch(model, tr_l, opt, crit, device, is_tumor)
        val_acc,_,_  = eval_epoch(model, val_l, device, is_tumor); sch.step()
        print(f"  [{name}] {ep:3d}/{n_epochs} loss={loss:.4f} train={tr_acc:.3f} val={val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
    print(f"  [{name}] Best val acc: {best:.4f}")


# =============================================================================
# GRAD-CAM  (visual explainability for the tumor classifier)
# =============================================================================
# Registers hooks on the last conv layer to capture activations and gradients.
# The heatmap shows which MRI regions drove the tumor prediction —
# essential for clinical interpretability.

class GradCAM:
    def __init__(self, model, target_layer):
        self.act, self.grad = None, None
        target_layer.register_forward_hook(      lambda m,i,o: setattr(self,'act', o.detach()))
        target_layer.register_full_backward_hook(lambda m,i,o: setattr(self,'grad',o[0].detach()))

    @torch.no_grad()
    def generate(self, logits, class_idx):
        with torch.enable_grad():
            logits[:, class_idx].backward(retain_graph=True)
        w   = self.grad.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1).squeeze()).cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)

    # ── STEP 1: FER2013 face emotion CNN ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: FER2013 — JPEG image folders → emotion CNN")
    print("="*60)
    fer_tr, fer_val, fer_test, fer_map = build_fer_loaders(
        CFG["fer_train_dir"], CFG["fer_test_dir"], CFG["batch_size"])
    fer_model = EmotionCNN2D(n_classes=CFG["n_fer_classes"]).to(DEVICE)
    train_model(fer_model, fer_tr, fer_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fer_cnn")

    # ── STEP 2: fMRI volumetric emotion CNN ──────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: fMRI — NIfTI 4D volumes → 3D emotion CNN")
    print("="*60)
    fmri_tr, fmri_val, n_fmri = build_fmri_loaders(CFG["fmri_root"], CFG["batch_size"])
    fmri_model = fMRIEmotionCNN3D(n_classes=n_fmri).to(DEVICE)
    train_model(fmri_model, fmri_tr, fmri_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fmri_cnn")

    # ── STEP 3: EEG band-power MLP ───────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: EEG — DEAP .dat / MNE .npy → band-power MLP")
    print("="*60)
    eeg_tr, eeg_val, feat_dim, n_eeg = build_eeg_loaders(
        CFG["deap_dir"], CFG["mne_dir"], CFG["batch_size"])
    eeg_model = EEGEmotionMLP(n_classes=n_eeg, feat_dim=feat_dim).to(DEVICE)
    train_model(eeg_model, eeg_tr, eeg_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="eeg_mlp")

    # ── STEP 4: Generate emotion embeddings for tumor MRI samples ────────────
    print("\n" + "="*60)
    print("STEP 4: Run FER model on tumor MRI images → emotion vectors")
    print("  (Emotion context is approximated from visual features of the scan)")
    print("="*60)
    fer_model.load_state_dict(torch.load("checkpoints/fer_cnn_best.pt", map_location=DEVICE))
    fer_model.eval()

    # Load tumor images at FER resolution (48×48) to pass through FER model
    probe_ds = BrainTumorDataset(
        CFG["tumor_dir"],
        transform=T.Compose([
            T.Resize((48,48)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        n_emotions=CFG["n_fer_classes"]
    )
    probe_loader   = DataLoader(probe_ds, CFG["batch_size"], shuffle=False)
    emotion_vectors = {}

    with torch.no_grad():
        for b_idx, (imgs, labels, _) in enumerate(probe_loader):
            probs = F.softmax(fer_model(imgs.to(DEVICE)), dim=-1).cpu().numpy()
            for j in range(len(labels)):
                si = b_idx * CFG["batch_size"] + j
                if si < len(probe_ds):
                    fname = os.path.basename(probe_ds.paths[si])
                    emotion_vectors[fname] = probs[j]   # shape (7,)

    print(f"  Emotion vectors generated for {len(emotion_vectors):,} tumor samples.")

    # ── STEP 5: Train emotion-conditioned tumor classifier ───────────────────
    print("\n" + "="*60)
    print("STEP 5: Brain Tumor — MRI + emotion embedding → tumor classifier")
    print("="*60)
    tumor_tr, tumor_val, tumor_ds = build_tumor_loaders(
        CFG["tumor_dir"], CFG["batch_size"],
        emotion_vectors=emotion_vectors, n_emotions=CFG["n_fer_classes"]
    )
    tumor_model = EmotionConditionedTumorCNN(
        n_tumor_classes=CFG["n_tumor_classes"],
        emotion_dim=CFG["n_fer_classes"]
    ).to(DEVICE)
    train_model(tumor_model, tumor_tr, tumor_val,
                CFG["epochs_tumor"], CFG["lr"], DEVICE, is_tumor=True, name="tumor_cnn")

    # ── STEP 6: Final evaluation ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Final Evaluation + Grad-CAM")
    print("="*60)
    tumor_model.load_state_dict(torch.load("checkpoints/tumor_cnn_best.pt", map_location=DEVICE))
    acc, y_true, y_pred = eval_epoch(tumor_model, tumor_val, DEVICE, is_tumor=True)
    print(f"\nTumor Classifier — Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=tumor_ds.class_names))

    # Grad-CAM on first sample
    tumor_model.eval()
    last_conv = list(list(tumor_model.mri_encoder.children())[-1].children())[-1].conv3
    cam_gen   = GradCAM(tumor_model, last_conv)
    img_t, lbl, e_vec = tumor_ds[0]
    out = tumor_model(img_t.unsqueeze(0).to(DEVICE), e_vec.unsqueeze(0).to(DEVICE))
    cam = cam_gen.generate(out, out.argmax().item())

    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    raw  = np.clip((img_t.permute(1,2,0).numpy()*std + mean)*255, 0, 255).astype(np.uint8)
    import cv2
    cam_r = cv2.resize(cam, (raw.shape[1], raw.shape[0]))
    heat  = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw, 0.6, heat, 0.4, 0)
    fig, ax = plt.subplots(1,3,figsize=(10,4))
    ax[0].imshow(raw);         ax[0].set_title("Original MRI"); ax[0].axis("off")
    ax[1].imshow(cam_r,cmap='jet'); ax[1].set_title("Grad-CAM");    ax[1].axis("off")
    ax[2].imshow(overlay);     ax[2].set_title("Overlay");     ax[2].axis("off")
    pred_name = tumor_ds.class_names[out.argmax().item()]
    true_name = tumor_ds.class_names[lbl]
    plt.suptitle(f"Predicted: {pred_name}  |  True: {true_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/gradcam_tumor_sample.png", dpi=150)
    plt.close()

    print("\n[Done] Checkpoints  → ./checkpoints/")
    print("[Done] Grad-CAM     → ./outputs/gradcam_tumor_sample.png")


if __name__ == "__main__":
    main()