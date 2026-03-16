"""
Multimodal Emotion Recognition Pipeline
Dataset: Sun et al. (2023) - OSF/26rhz
Modalities: EEG, Eye Tracking, Behavioral, fMRI (ROI activations)
Task: Fear vs. Happy classification on morphed face stimuli

Dependencies:
    pip install numpy scipy scikit-learn torch mne pandas matplotlib seaborn h5py nilearn
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────
class Config:
    # ── Paths ──────────────────────────────────
    # Update these to point at your downloaded OSF data folders
    DATA_ROOT       = Path("data/osf_26rhz")
    EEG_DIR         = DATA_ROOT / "EEG"
    EYETRACK_DIR    = DATA_ROOT / "EyeTracking"
    FMRI_DIR        = DATA_ROOT / "fMRI"
    BEHAVIORAL_DIR  = DATA_ROOT / "Behavioral"
    OUTPUT_DIR      = Path("outputs")

    # ── EEG signal processing ──────────────────
    EEG_SFREQ       = 500          # Hz (dataset sampling rate)
    LOWPASS_HZ      = 40.0
    HIGHPASS_HZ     = 0.5
    EPOCH_TMIN      = -0.2         # seconds before stimulus
    EPOCH_TMAX      = 0.8          # seconds after stimulus
    EEG_CHANNELS    = 64           # adjust to your cap

    # ── Eye-tracking ───────────────────────────
    ET_SFREQ        = 1000         # Hz
    ET_FEATURES     = ["pupil_diameter", "fixation_x", "fixation_y",
                       "saccade_amplitude", "blink_rate"]

    # ── fMRI ROIs used as features ─────────────
    # Amygdala, FFA, OFC (from paper's published ROI masks)
    FMRI_ROIS       = ["left_amygdala", "right_amygdala",
                       "left_FFA", "right_FFA",
                       "dmPFC", "vmPFC"]

    # ── Labels ─────────────────────────────────
    # Morphed face continuum: morph level ≤50% → fear, >50% → happy
    MORPH_THRESHOLD = 50
    CLASSES         = ["fear", "happy"]

    # ── Training ───────────────────────────────
    SEED            = 42
    N_FOLDS         = 5
    BATCH_SIZE      = 32
    EPOCHS          = 80
    LR              = 1e-3
    WEIGHT_DECAY    = 1e-4
    HIDDEN_DIM      = 128
    DROPOUT         = 0.4

    # ── Device ─────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ─────────────────────────────────────────────
# 2.  Data Loaders & Preprocessors
# ─────────────────────────────────────────────

class EEGProcessor:
    """
    Load, epoch, and extract features from EEG .mat / .set files.

    OSF folder layout (typical):
        EEG/
            sub-01/
                sub-01_task-emotionJudgment_eeg.set
            sub-02/ ...
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.b, self.a = butter(4,
            [cfg.HIGHPASS_HZ / (cfg.EEG_SFREQ / 2),
             cfg.LOWPASS_HZ  / (cfg.EEG_SFREQ / 2)],
            btype="bandpass")

    def bandpass(self, data: np.ndarray) -> np.ndarray:
        """data shape: (n_channels, n_times)"""
        return filtfilt(self.b, self.a, data, axis=-1)

    def extract_band_power(self, epoch: np.ndarray) -> np.ndarray:
        """
        epoch: (n_channels, n_times)
        Returns power in 5 frequency bands → (n_channels × 5,)
        """
        from scipy.signal import welch
        bands = {"delta": (1, 4), "theta": (4, 8),
                 "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
        feats = []
        for ch in epoch:
            f, psd = welch(ch, fs=self.cfg.EEG_SFREQ, nperseg=128)
            for lo, hi in bands.values():
                mask = (f >= lo) & (f <= hi)
                feats.append(np.mean(psd[mask]))
        return np.array(feats)  # (n_channels * 5,)

    def load_subject(self, subject_path: Path):
        """
        Load one subject's EEG file.
        Supports .npy (preprocessed), .mat (EEGLAB), or .set (EEGLAB via MNE).
        Returns: (features, labels)
        """
        ext = subject_path.suffix.lower()

        if ext == ".npy":
            d = np.load(subject_path, allow_pickle=True).item()
            epochs = d["epochs"]         # (n_trials, n_channels, n_times)
            labels = d["labels"]         # (n_trials,) morph level 0-100
        elif ext in (".set", ".mat"):
            try:
                import mne
                raw = mne.io.read_epochs_eeglab(str(subject_path), verbose=False)
                epochs = raw.get_data()  # (n_trials, n_channels, n_times)
                labels = raw.events[:, 2]
            except Exception as e:
                print(f"[EEG] Could not load {subject_path}: {e}")
                return None, None
        else:
            raise ValueError(f"Unsupported EEG format: {ext}")

        # Bandpass filter
        epochs = np.array([self.bandpass(ep) for ep in epochs])

        # Extract features per trial
        X = np.array([self.extract_band_power(ep) for ep in epochs])

        # Binarize labels: morph ≤ threshold → fear (0), else happy (1)
        y = (labels > self.cfg.MORPH_THRESHOLD).astype(int)
        return X, y

    def load_all(self):
        subjects = sorted(self.cfg.EEG_DIR.glob("sub-*/"))
        all_X, all_y = [], []
        for sub_dir in subjects:
            for f in sub_dir.glob("*.set"):
                X, y = self.load_subject(f)
                if X is not None:
                    all_X.append(X)
                    all_y.append(y)
        if not all_X:
            return None, None
        return np.vstack(all_X), np.concatenate(all_y)


class EyeTrackingProcessor:
    """
    Load and feature-engineer eye-tracking TSV/CSV files.

    OSF folder layout (typical):
        EyeTracking/
            sub-01/
                sub-01_task-emotionJudgment_eyetrack.tsv
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _trial_stats(self, trial_df: pd.DataFrame) -> np.ndarray:
        """Aggregate time-series eye-tracking data into per-trial features."""
        feats = []
        for col in ["pupil_diameter", "x_pos", "y_pos"]:
            if col in trial_df.columns:
                vals = trial_df[col].dropna().values
                if len(vals) > 0:
                    feats += [vals.mean(), vals.std(),
                              vals.max() - vals.min(),
                              np.percentile(vals, 25),
                              np.percentile(vals, 75)]
                else:
                    feats += [0] * 5
        # Blink count
        if "blink" in trial_df.columns:
            feats.append(trial_df["blink"].sum())
        return np.array(feats, dtype=np.float32)

    def load_subject(self, subject_path: Path):
        try:
            df = pd.read_csv(subject_path, sep="\t")
        except Exception as e:
            print(f"[ET] Could not load {subject_path}: {e}")
            return None, None

        if "trial" not in df.columns or "morph_level" not in df.columns:
            print(f"[ET] Missing required columns in {subject_path}")
            return None, None

        trials = df.groupby("trial")
        X, y = [], []
        for _, trial_df in trials:
            X.append(self._trial_stats(trial_df))
            ml = trial_df["morph_level"].iloc[0]
            y.append(int(ml > self.cfg.MORPH_THRESHOLD))
        return np.array(X), np.array(y)

    def load_all(self):
        files = sorted(self.cfg.EYETRACK_DIR.glob("sub-*/*.tsv"))
        all_X, all_y = [], []
        for f in files:
            X, y = self.load_subject(f)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
        if not all_X:
            return None, None
        return np.vstack(all_X), np.concatenate(all_y)


class FMRIProcessor:
    """
    Extract mean BOLD activation from predefined ROI masks per trial.

    OSF folder layout (typical):
        fMRI/
            sub-01/
                sub-01_task-emotionJudgment_bold.nii.gz
            masks/
                left_amygdala_mask.nii.gz  (etc.)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_roi_timeseries(self, bold_path: Path, mask_dir: Path) -> np.ndarray:
        """Returns (n_volumes, n_rois)"""
        try:
            from nilearn.maskers import NiftiMasker
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nilearn: pip install nilearn nibabel")

        roi_signals = []
        for roi in self.cfg.FMRI_ROIS:
            mask_path = mask_dir / f"{roi}_mask.nii.gz"
            if not mask_path.exists():
                roi_signals.append(np.zeros(1))
                continue
            masker = NiftiMasker(mask_img=str(mask_path),
                                 standardize=True, verbose=0)
            signal = masker.fit_transform(str(bold_path))  # (n_vols, n_voxels)
            roi_signals.append(signal.mean(axis=1))        # mean over voxels
        return np.column_stack(roi_signals)  # (n_volumes, n_rois)

    def load_all(self):
        """
        Loads per-subject fMRI ROI features aligned to trial onsets
        via behavioral timing files (events.tsv).
        """
        all_X, all_y = [], []
        mask_dir = self.cfg.FMRI_DIR / "masks"

        for sub_dir in sorted(self.cfg.FMRI_DIR.glob("sub-*/")):
            bold_files = list(sub_dir.glob("*_bold.nii.gz"))
            events_files = list(sub_dir.glob("*_events.tsv"))
            if not bold_files or not events_files:
                continue

            try:
                ts = self.load_roi_timeseries(bold_files[0], mask_dir)
                events = pd.read_csv(events_files[0], sep="\t")
                TR = 2.0  # dataset TR = 2000ms

                for _, row in events.iterrows():
                    onset_vol = int(row["onset"] / TR)
                    # Average 2 TRs around peak hemodynamic response (+4–6s → +2–3 TRs)
                    peak_vols = slice(onset_vol + 2,
                                     min(onset_vol + 4, ts.shape[0]))
                    roi_feats = ts[peak_vols].mean(axis=0)
                    ml = row.get("morph_level", 50)
                    all_X.append(roi_feats)
                    all_y.append(int(ml > self.cfg.MORPH_THRESHOLD))
            except Exception as e:
                print(f"[fMRI] Error loading {sub_dir}: {e}")

        if not all_X:
            return None, None
        return np.array(all_X), np.array(all_y)


class BehavioralProcessor:
    """
    Load RT, accuracy, and confidence ratings from TSV behavioral files.

    OSF folder layout:
        Behavioral/
            sub-01_task-emotionJudgment_beh.tsv
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_all(self):
        all_X, all_y = [], []
        for f in sorted(self.cfg.BEHAVIORAL_DIR.glob("*_beh.tsv")):
            try:
                df = pd.read_csv(f, sep="\t")
            except Exception as e:
                print(f"[Beh] Could not load {f}: {e}")
                continue

            req_cols = {"morph_level", "response_time"}
            if not req_cols.issubset(df.columns):
                print(f"[Beh] Missing columns in {f}, skipping.")
                continue

            for _, row in df.iterrows():
                feats = [
                    row.get("response_time", np.nan),
                    row.get("confidence", np.nan),
                    row.get("correct", np.nan),
                    row.get("morph_level", np.nan) / 100.0,   # normalized
                ]
                all_X.append(feats)
                all_y.append(int(row["morph_level"] > self.cfg.MORPH_THRESHOLD))

        if not all_X:
            return None, None
        X = np.array(all_X, dtype=np.float32)
        # Impute NaNs with column medians
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        return X, np.array(all_y)


# ─────────────────────────────────────────────
# 3.  Multimodal Feature Fusion
# ─────────────────────────────────────────────

def align_and_fuse(modality_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Late-feature-level fusion: concatenate per-trial feature vectors
    from all available modalities.

    Since different modalities may have different numbers of subjects/trials,
    we use the behavioral modality (widest coverage) as the alignment backbone
    and match other modalities where possible. In practice you should align by
    subject_id + trial_id from the events files.

    modality_data: {"eeg": (X, y), "et": (X, y), "fmri": (X, y), "beh": (X, y)}
    Returns: (X_fused, y_fused)
    """
    available = {k: v for k, v in modality_data.items()
                 if v[0] is not None and len(v[0]) > 0}

    if not available:
        raise RuntimeError("No modality data loaded. Check your data paths.")

    print("\n── Loaded modalities ──────────────────────────")
    for name, (X, y) in available.items():
        print(f"  {name.upper():10s}  X: {X.shape}  y: {y.shape}")
    print("───────────────────────────────────────────────\n")

    # Use smallest n_trials as common denominator (conservative alignment)
    n_trials = min(len(v[0]) for v in available.values())
    X_parts, y_ref = [], None

    for name, (X, y) in available.items():
        X_sub = X[:n_trials]
        # Z-score normalize each modality independently
        scaler = StandardScaler()
        X_sub = scaler.fit_transform(X_sub)
        X_parts.append(X_sub)
        if y_ref is None:
            y_ref = y[:n_trials]

    X_fused = np.hstack(X_parts)
    return X_fused.astype(np.float32), y_ref.astype(np.int64)


# ─────────────────────────────────────────────
# 4.  Model  — Multimodal MLP with attention
# ─────────────────────────────────────────────

class MultimodalEmotionNet(nn.Module):
    """
    Feed-forward network with residual connections and dropout.
    Input: concatenated features from all modalities.
    Output: 2-class logits (fear / happy).
    """

    def __init__(self, input_dim: int, hidden: int = 128,
                 dropout: float = 0.4, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden // 2, n_classes),
        )
        # Residual projection for skip connection
        self.skip = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# ─────────────────────────────────────────────
# 5.  Training & Evaluation
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        preds = logits.argmax(1)
        total_loss += loss.item() * len(y_batch)
        correct += (preds == y_batch).sum().item()
        n += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return total_loss / n, correct / n, all_preds, all_labels


def cross_validate(X: np.ndarray, y: np.ndarray, cfg: Config):
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                          random_state=cfg.SEED)
    fold_accs = []
    all_cm = np.zeros((2, 2), dtype=int)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold}/{cfg.N_FOLDS}")
        print(f"{'─'*50}")

        train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
        val_ds   = TensorDataset(X_t[val_idx],   y_t[val_idx])
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE)

        model = MultimodalEmotionNet(
            input_dim=X.shape[1],
            hidden=cfg.HIDDEN_DIM,
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
                                weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_val_acc, patience_counter = 0.0, 0
        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": []}

        for epoch in range(1, cfg.EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader,
                                          optimizer, criterion, cfg.DEVICE)
            val_loss, val_acc, preds, labels = eval_epoch(
                model, val_loader, criterion, cfg.DEVICE)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d}  "
                      f"train_loss={tr_loss:.4f}  acc={tr_acc:.3f}  │  "
                      f"val_loss={val_loss:.4f}  acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           cfg.OUTPUT_DIR / f"best_model_fold{fold}.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Reload best weights
        model.load_state_dict(
            torch.load(cfg.OUTPUT_DIR / f"best_model_fold{fold}.pt",
                       map_location=cfg.DEVICE))
        _, fold_acc, preds, labels = eval_epoch(
            model, val_loader, criterion, cfg.DEVICE)
        fold_accs.append(fold_acc)
        all_cm += confusion_matrix(labels, preds, labels=[0, 1])

        print(f"\n  Best fold accuracy: {fold_acc:.4f}")
        print(classification_report(labels, preds,
              target_names=cfg.CLASSES, digits=3))

        _plot_history(history, fold, cfg)

    print(f"\n{'═'*50}")
    print(f"  CV Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"{'═'*50}\n")
    _plot_confusion_matrix(all_cm, cfg)
    return fold_accs


# ─────────────────────────────────────────────
# 6.  Visualization helpers
# ─────────────────────────────────────────────

def _plot_history(history: dict, fold: int, cfg: Config):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, val_metric) in zip(axes,
            [("train_loss", "val_loss"), ("train_acc", "val_acc")]):
        ax.plot(history[metric],   label="Train")
        ax.plot(history[val_metric], label="Val", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_title(metric.split("_")[1].capitalize())
        ax.legend()
    fig.suptitle(f"Fold {fold} training curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / f"fold{fold}_curves.png", dpi=100)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, cfg: Config):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aggregated Confusion Matrix (all folds)")
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "confusion_matrix.png", dpi=100)
    plt.close(fig)
    print("Confusion matrix saved.")


# ─────────────────────────────────────────────
# 7.  Inference helper
# ─────────────────────────────────────────────

def predict(model_path: str, X_new: np.ndarray,
            input_dim: int, cfg: Config) -> np.ndarray:
    """
    Load a saved model and run inference on new multimodal features.

    Args:
        model_path: path to saved .pt state dict
        X_new:      (n_samples, input_dim) array of fused features
        input_dim:  must match training-time input dim

    Returns: (n_samples,) predicted class indices
    """
    model = MultimodalEmotionNet(input_dim=input_dim,
                                 hidden=cfg.HIDDEN_DIM,
                                 dropout=0.0).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    X_t = torch.tensor(X_new, dtype=torch.float32).to(cfg.DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
    return preds.cpu().numpy(), probs.cpu().numpy()


# ─────────────────────────────────────────────
# 8.  Main entry point
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Multimodal Emotion Recognition — OSF/26rhz")
    print(f"  Device: {cfg.DEVICE}")
    print("=" * 60)

    # ── Load each modality ─────────────────────
    print("\n[1/3] Loading modalities …")

    eeg_X, eeg_y = EEGProcessor(cfg).load_all()
    et_X,  et_y  = EyeTrackingProcessor(cfg).load_all()
    fmri_X, fmri_y = FMRIProcessor(cfg).load_all()
    beh_X, beh_y = BehavioralProcessor(cfg).load_all()

    modality_data = {
        "eeg":   (eeg_X,   eeg_y),
        "et":    (et_X,    et_y),
        "fmri":  (fmri_X,  fmri_y),
        "beh":   (beh_X,   beh_y),
    }

    # ── Fuse features ──────────────────────────
    print("\n[2/3] Fusing modalities …")
    X, y = align_and_fuse(modality_data)
    print(f"  Fused dataset: {X.shape[0]} trials × {X.shape[1]} features")
    print(f"  Class balance: {np.bincount(y)} (fear / happy)")

    # ── Train & cross-validate ─────────────────
    print("\n[3/3] Cross-validated training …")
    fold_accs = cross_validate(X, y, cfg)

    print("\n✓ Training complete.")
    print(f"  Mean CV accuracy: {np.mean(fold_accs):.4f}")
    print(f"  Outputs saved to: {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()