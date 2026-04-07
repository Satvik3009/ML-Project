"""
main1.py — CORRECTED PATH VERSION
=============================================================================
FULL ML PIPELINE: Multi-Modal Emotion Recognition → Brain Tumor Detection
=============================================================================
FIXES vs original:
  - data/fer/train      → data/fer2013/train       (was completely wrong folder)
  - data/fer/test       → data/fer2013/test
  - data/MNE-sample-data → data/MNE-sample-data/MEG/sample  (missing subdirs)
  - data/dataset/data   → data/figshare_brain       (was completely wrong folder)
  - num_workers=4       → num_workers=0             (Windows multiprocessing fix)
  - All paths now imported from paths.py (no hardcoded absolutes)
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

# ── Paths — imported from central config (no hardcoded strings) ───────────────
from paths import P

P.makedirs()

CFG = {
    # FIXED: was "data/fer/train" and "data/fer/test"
    "fer_train_dir"  : str(P.FER_TRAIN),
    "fer_test_dir"   : str(P.FER_TEST),
    # FIXED: was "data/fmri" (still correct but now uses P.FMRI)
    "fmri_root"      : str(P.FMRI),
    # deap_dir: points to parent folder; EEGEmotionDataset handles MIDI-only fallback
    "deap_dir"       : str(P.DEAP),
    # FIXED: was "data/MNE-sample-data" — missing /MEG/sample subdirectory
    "mne_dir"        : str(P.MNE),
    # FIXED: was "data/dataset/data" — completely wrong folder name
    "tumor_dir"      : str(P.TUMOR),
    "batch_size"     : 32,
    "epochs_emotion" : 20,
    "epochs_tumor"   : 15,
    "lr"             : 1e-3,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    # FIXED: was 4 — crashes on Windows with DataLoader multiprocessing
    "num_workers"    : 0,
    "n_fer_classes"  : 7,
    "n_tumor_classes": 4,
    "tumor_classes"  : ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
}
DEVICE = torch.device(CFG["device"])
print(f"[Config] Device: {DEVICE}")
print(f"[Config] FER train : {CFG['fer_train_dir']}")
print(f"[Config] MNE dir   : {CFG['mne_dir']}")
print(f"[Config] Tumor dir : {CFG['tumor_dir']}")


# =============================================================================
# DATASET 1 — FER2013
# =============================================================================

def build_fer_loaders(train_dir, test_dir, batch_size, nw=0):
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((48, 48)),
        T.RandomHorizontalFlip(p=0.5),
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
    n_val     = len(full_test) // 2
    val_ds, test_ds = random_split(full_test, [n_val, len(full_test)-n_val],
                                   generator=torch.Generator().manual_seed(42))

    print(f"[FER2013] train={len(train_ds):,}  val={n_val:,}")
    print(f"[FER2013] labels: {train_ds.class_to_idx}")

    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=nw, pin_memory=False),
            DataLoader(val_ds,   batch_size, shuffle=False, num_workers=nw),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=nw),
            train_ds.class_to_idx)


# =============================================================================
# DATASET 2 — fMRI
# =============================================================================

class FMRIEmotionDataset(Dataset):
    def __init__(self, fmri_root, target_shape=(32, 32, 20)):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("pip install nibabel")

        self.volumes, self.labels = [], []
        raw_labels = []

        for subj in sorted(Path(fmri_root).iterdir()):
            if not subj.is_dir():
                continue
            niis = (list(subj.glob("func/*.nii.gz")) + list(subj.glob("func/*.nii")) +
                    list(subj.glob("*.nii.gz"))       + list(subj.glob("*.nii")))
            lbls = (list(subj.glob("labels.csv"))     + list(subj.glob("func/labels.csv")))
            if not niis or not lbls:
                print(f"  [fMRI] skipping {subj.name} — no .nii or labels.csv")
                continue
            data4d = nib.load(str(niis[0])).get_fdata()
            df     = pd.read_csv(lbls[0])
            for _, row in df.iterrows():
                tr = int(row['TR'])
                if tr >= data4d.shape[3]: continue
                self.volumes.append(self._process(data4d[:,:,:,tr], target_shape))
                raw_labels.append(str(row['emotion']).strip().lower())

        if not self.volumes:
            raise ValueError(f"No fMRI samples loaded from {fmri_root}")

        le = LabelEncoder()
        self.labels    = le.fit_transform(raw_labels)
        self.n_classes = len(le.classes_)
        print(f"[fMRI] {len(self.volumes):,} volumes | classes={list(le.classes_)}")

    def _process(self, vol, target):
        from scipy.ndimage import zoom
        factors = [t/s for t,s in zip(target, vol.shape)]
        v = zoom(vol, factors, order=1).astype(np.float32)
        return (v - v.mean()) / (v.std() + 1e-8)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.volumes[i]).unsqueeze(0),
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def build_fmri_loaders(fmri_root, batch_size, nw=0):
    ds    = FMRIEmotionDataset(fmri_root)
    n_val = int(0.2 * len(ds))
    tr_ds, v_ds = random_split(ds, [len(ds)-n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr_ds, batch_size, shuffle=True,  num_workers=nw),
            DataLoader(v_ds,  batch_size, shuffle=False, num_workers=nw),
            ds.n_classes)


# =============================================================================
# DATASET 3 — EEG
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
            raise FileNotFoundError(
                f"No EEG data found.\n"
                f"  DEAP: {deap_dir}\n"
                f"  MNE:  {mne_dir}\n"
                f"  Provide s*.dat files (DEAP) or epochs_data.npy (MNE).")

        self.features = np.stack(self.features).astype(np.float32)
        self.labels   = np.array(self.labels, dtype=np.int64)
        print(f"[EEG] features={self.features.shape}  n_classes={len(np.unique(self.labels))}")

    def _extract(self, trial):
        out = []
        for lo, hi in self.BANDS.values():
            nyq  = self.fs / 2.0
            b, a = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype='band')
            filt = scipy.signal.filtfilt(b, a, trial, axis=-1)
            _, psd = scipy.signal.welch(filt, fs=self.fs, nperseg=256, axis=-1)
            out.extend([psd.mean(axis=-1),
                        0.5*np.log(2*np.pi*np.e*np.var(filt,axis=-1)+1e-8)])
        return np.concatenate(out)

    def _load_deap(self, deap_dir):
        for path in sorted(Path(deap_dir).glob("s*.dat")):
            with open(path,'rb') as f:
                s = pickle.load(f, encoding='latin1')
            eeg    = s['data'][:, :32, 3*self.fs:]
            labels = s['labels']
            for t in range(eeg.shape[0]):
                self.features.append(self._extract(eeg[t]))
                v, a = labels[t,0], labels[t,1]
                self.labels.append(1 if (v>=5 and a>=5) else 0)

    def _load_mne(self, mne_dir):
        data   = np.load(Path(mne_dir)/"epochs_data.npy")
        labels = np.load(Path(mne_dir)/"epochs_labels.npy")
        for i in range(len(data)):
            self.features.append(self._extract(data[i]))
            self.labels.append(int(labels[i]))

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.features[i]), torch.tensor(int(self.labels[i])))


def build_eeg_loaders(deap_dir, mne_dir, batch_size, nw=0):
    ds = EEGEmotionDataset(deap_dir=deap_dir, mne_dir=mne_dir)
    feat_dim  = ds.features.shape[1]
    n_classes = len(np.unique(ds.labels))
    tr_i, v_i = train_test_split(range(len(ds)), test_size=0.2,
                                  stratify=ds.labels, random_state=42)
    return (DataLoader(Subset(ds,tr_i), batch_size, shuffle=True,  num_workers=nw),
            DataLoader(Subset(ds,v_i),  batch_size, shuffle=False, num_workers=nw),
            feat_dim, n_classes)


# =============================================================================
# DATASET 4 — Brain Tumor (FigShare .mat files via h5py)
# =============================================================================

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None, emotion_vectors=None, n_emotions=7):
        self.paths, self.labels, raw_labels = [], [], []
        self.transform       = transform
        self.emotion_vectors = emotion_vectors or {}
        self.n_emotions      = n_emotions

        # FigShare structure: figshare_brain/data/1.mat … 3064.mat
        data_dir = Path(root_dir) / "data"
        if data_dir.exists():
            # HDF5 .mat files (MATLAB v7.3)
            self._load_mat_files(data_dir)
        else:
            # Fallback: class-named JPEG subfolders
            for cls_dir in sorted(Path(root_dir).iterdir()):
                if not cls_dir.is_dir(): continue
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    self.paths.append(str(img_path))
                    raw_labels.append(cls_dir.name.lower())
            le = LabelEncoder()
            self.labels    = le.fit_transform(raw_labels).tolist()
            self.class_names = list(le.classes_)

        print(f"[Tumor] {len(self.paths):,} images | classes={self.class_names}")

    def _load_mat_files(self, data_dir):
        """Load FigShare HDF5 .mat files."""
        import h5py
        remap = {1: 1, 2: 0, 3: 2}  # mat label -> 0-based alphabetical
        self.class_names = ['glioma', 'meningioma', 'pituitary']
        for mat_path in sorted(data_dir.glob("*.mat"), key=lambda p: int(p.stem)):
            try:
                with h5py.File(str(mat_path), 'r') as f:
                    lbl = int(np.array(f['cjdata/label']).flat[0])
                self.paths.append(str(mat_path))
                self.labels.append(remap.get(lbl, 0))
            except Exception:
                continue

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        if path.endswith(".mat"):
            import h5py
            with h5py.File(path, 'r') as f:
                arr = np.array(f['cjdata/image'], dtype=np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn: arr = (arr - mn) / (mx - mn) * 255.0
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")

        if self.transform: img = self.transform(img)
        fname = os.path.basename(path)
        e_vec = self.emotion_vectors.get(fname, np.zeros(self.n_emotions, dtype=np.float32))
        return img, int(self.labels[i]), torch.tensor(e_vec, dtype=torch.float32)


def build_tumor_loaders(tumor_dir, batch_size, emotion_vectors=None, n_emotions=7, nw=0):
    train_tf = T.Compose([
        T.Resize((128,128)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(15), T.ColorJitter(brightness=0.15, contrast=0.2),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((128,128)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    full  = BrainTumorDataset(tumor_dir, train_tf, emotion_vectors, n_emotions)
    val_d = BrainTumorDataset(tumor_dir, eval_tf,  emotion_vectors, n_emotions)
    tr_i, v_i = train_test_split(range(len(full)), test_size=0.2,
                                  stratify=full.labels, random_state=42)
    return (DataLoader(Subset(full,  tr_i), batch_size, shuffle=True,  num_workers=nw),
            DataLoader(Subset(val_d, v_i),  batch_size, shuffle=False, num_workers=nw),
            full)


# =============================================================================
# MODELS
# =============================================================================

class EmotionCNN2D(nn.Module):
    def __init__(self, n_classes, dropout=0.4):
        super().__init__()
        base = tv_models.resnet18(weights="IMAGENET1K_V1")
        self.feature_dim = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )
        self.net, self._emb = base, None
        self.net.avgpool.register_forward_hook(
            lambda m,i,o: setattr(self, '_emb', o.flatten(1)))
    def forward(self, x):         return self.net(x)
    def get_embedding(self, x):   self.forward(x); return self._emb


class fMRIEmotionCNN3D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1),nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64*8, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):       return self.classifier(self.encoder(x))
    def get_embedding(self, x): return self.encoder(x).flatten(1)


class EEGEmotionMLP(nn.Module):
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
    def forward(self, x):       return self.net(x)
    def get_embedding(self, x): self.forward(x); return self._emb


class EmotionConditionedTumorCNN(nn.Module):
    def __init__(self, n_tumor_classes=4, emotion_dim=7):
        super().__init__()
        base = tv_models.resnet50(weights="IMAGENET1K_V1")
        self.mri_encoder     = nn.Sequential(*list(base.children())[:-1])
        self.emotion_encoder = nn.Sequential(nn.Linear(emotion_dim,64), nn.ReLU(), nn.Linear(64,64))
        self.classifier      = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048+64, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_tumor_classes),
        )
    def forward(self, img, emotion_vec):
        mri = self.mri_encoder(img).flatten(1)
        emo = self.emotion_encoder(emotion_vec)
        return self.classifier(torch.cat([mri, emo], dim=1))


# =============================================================================
# TRAINING
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
        total_loss += loss.item()*y.size(0)
        correct    += (logits.argmax(1)==y).sum().item()
        n          += y.size(0)
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
    ckpt_path = str(P.CKPT / f"{name}_best.pt")
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, n_epochs+1):
        loss, tr_acc = train_epoch(model, tr_l, opt, crit, device, is_tumor)
        val_acc,_,_  = eval_epoch(model, val_l, device, is_tumor)
        sch.step()
        print(f"  [{name}] {ep:3d}/{n_epochs} loss={loss:.4f} train={tr_acc:.3f} val={val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), ckpt_path)
    print(f"  [{name}] Best val: {best:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    nw = CFG["num_workers"]

    print("\n" + "="*60)
    print("STEP 1: FER2013")
    print("="*60)
    fer_tr, fer_val, fer_test, fer_map = build_fer_loaders(
        CFG["fer_train_dir"], CFG["fer_test_dir"], CFG["batch_size"], nw)
    fer_model = EmotionCNN2D(n_classes=CFG["n_fer_classes"]).to(DEVICE)
    train_model(fer_model, fer_tr, fer_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fer_cnn")

    print("\n" + "="*60)
    print("STEP 2: fMRI")
    print("="*60)
    fmri_tr, fmri_val, n_fmri = build_fmri_loaders(CFG["fmri_root"], CFG["batch_size"], nw)
    fmri_model = fMRIEmotionCNN3D(n_classes=n_fmri).to(DEVICE)
    train_model(fmri_model, fmri_tr, fmri_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fmri_cnn")

    print("\n" + "="*60)
    print("STEP 3: EEG")
    print("="*60)
    eeg_tr, eeg_val, feat_dim, n_eeg = build_eeg_loaders(
        CFG["deap_dir"], CFG["mne_dir"], CFG["batch_size"], nw)
    eeg_model = EEGEmotionMLP(n_classes=n_eeg, feat_dim=feat_dim).to(DEVICE)
    train_model(eeg_model, eeg_tr, eeg_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="eeg_mlp")

    print("\n" + "="*60)
    print("STEP 4: Emotion vectors for tumor MRI")
    print("="*60)
    fer_ckpt = str(P.CKPT / "fer_cnn_best.pt")
    fer_model.load_state_dict(torch.load(fer_ckpt, map_location=DEVICE))
    fer_model.eval()

    probe_ds = BrainTumorDataset(
        CFG["tumor_dir"],
        transform=T.Compose([
            T.Resize((48,48)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        n_emotions=CFG["n_fer_classes"]
    )
    probe_loader    = DataLoader(probe_ds, CFG["batch_size"], shuffle=False, num_workers=nw)
    emotion_vectors = {}

    with torch.no_grad():
        for b_idx, (imgs, labels, _) in enumerate(probe_loader):
            probs = F.softmax(fer_model(imgs.to(DEVICE)), dim=-1).cpu().numpy()
            for j in range(len(labels)):
                si = b_idx * CFG["batch_size"] + j
                if si < len(probe_ds):
                    fname = os.path.basename(probe_ds.paths[si])
                    emotion_vectors[fname] = probs[j]

    print(f"  Emotion vectors: {len(emotion_vectors):,} samples")

    print("\n" + "="*60)
    print("STEP 5: Tumor Classifier")
    print("="*60)
    tumor_tr, tumor_val, tumor_ds = build_tumor_loaders(
        CFG["tumor_dir"], CFG["batch_size"],
        emotion_vectors=emotion_vectors, n_emotions=CFG["n_fer_classes"], nw=nw
    )
    tumor_model = EmotionConditionedTumorCNN(
        n_tumor_classes=CFG["n_tumor_classes"],
        emotion_dim=CFG["n_fer_classes"]
    ).to(DEVICE)
    train_model(tumor_model, tumor_tr, tumor_val,
                CFG["epochs_tumor"], CFG["lr"], DEVICE, is_tumor=True, name="tumor_cnn")

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    tumor_ckpt = str(P.CKPT / "tumor_cnn_best.pt")
    tumor_model.load_state_dict(torch.load(tumor_ckpt, map_location=DEVICE))
    acc, y_true, y_pred = eval_epoch(tumor_model, tumor_val, DEVICE, is_tumor=True)
    print(f"\nTumor Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=tumor_ds.class_names))
    print(f"\nCheckpoints → {P.CKPT}")


if __name__ == "__main__":
    main()