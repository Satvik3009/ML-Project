"""
paths.py — Single source of truth for all dataset and output paths.

Import this in every script:
    from paths import P

Paths are always relative to the project root (the folder containing this file).
No hardcoded absolute paths anywhere.

Actual folder names on disk (from .gitignore / te.py scan):
    data/deap-dataset/
    data/fer2013/          ← lowercase
    data/figshare_brain/
    data/fmri/
    data/MNE-sample-data/
"""

from pathlib import Path

# ── Project root = folder that contains this file ────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Data root ─────────────────────────────────────────────────────────────────
DATA = ROOT / "data"

class P:
    """Namespace for all project paths. Access like: P.FER_TRAIN"""

    # ── FER2013 ───────────────────────────────────────────────────────────────
    # Kaggle: msambare/fer2013
    # Structure: data/fer2013/train/{angry,disgust,...}/*.jpg
    FER_TRAIN = DATA / "fer2013" / "train"
    FER_TEST  = DATA / "fer2013" / "test"

    # ── DEAP EEG ──────────────────────────────────────────────────────────────
    # Academic download (not on Kaggle): s01.dat … s32.dat
    # Kaggle version has only MIDI audio — EEGEmotionDataset falls back to MNE.
    DEAP      = DATA / "deap-dataset"
    DEAP_DAT  = DATA / "deap-dataset" / "data_preprocessed_python"  # .dat files if present

    # ── MNE sample EEG/MEG ────────────────────────────────────────────────────
    # Kaggle: mubin986/mne-sample-data-processed
    # Raw .fif files live inside MEG/sample/
    MNE       = DATA / "MNE-sample-data" / "MEG" / "sample"

    # ── fMRI ──────────────────────────────────────────────────────────────────
    # Kaggle: irajahangari/fmri-dataset-for-emotion-recognition
    # Structure:
    #   data/fmri/Sub-01/*.nii(.gz)
    #   data/fmri/onsettime/*.tsv or *.mat
    # NOTE: folder is "onsettime" (10 letters), not "onsetime" (9).
    FMRI          = DATA / "fmri"
    FMRI_ONSET    = DATA / "fmri" / "onsettime"   # TSV / MAT label files

    # ── FigShare Brain Tumor ──────────────────────────────────────────────────
    # Kaggle: ashkhagan/figshare-brain-tumor-dataset
    # Structure:
    #   data/figshare_brain/cvind.mat
    #   data/figshare_brain/data/1.mat … 3064.mat
    TUMOR         = DATA / "figshare_brain"
    TUMOR_DATA    = DATA / "figshare_brain" / "data"
    TUMOR_CVIND   = DATA / "figshare_brain" / "cvind.mat"

    # ── Checkpoints & outputs ─────────────────────────────────────────────────
    CKPT    = ROOT / "checkpoints"
    OUTPUTS = ROOT / "outputs"

    # ── Helper: create output dirs on first use ───────────────────────────────
    @classmethod
    def makedirs(cls):
        for d in (cls.CKPT, cls.OUTPUTS):
            d.mkdir(parents=True, exist_ok=True)

    # ── Sanity check: print which data folders exist ──────────────────────────
    @classmethod
    def check(cls):
        checks = {
            "FER train":    cls.FER_TRAIN,
            "FER test":     cls.FER_TEST,
            "DEAP (pickle)":cls.DEAP_DAT,
            "MNE (.fif)":   cls.MNE,
            "fMRI":         cls.FMRI,
            "fMRI onset":   cls.FMRI_ONSET,
            "Tumor root":   cls.TUMOR,
            "Tumor data":   cls.TUMOR_DATA,
        }
        ok, missing = [], []
        for name, path in checks.items():
            (ok if path.exists() else missing).append((name, path))

        print("\n── Path check ──────────────────────────────────────")
        for name, path in ok:
            print(f"  ✓  {name:<20} {path}")
        for name, path in missing:
            print(f"  ✗  {name:<20} {path}  ← NOT FOUND")
        print("────────────────────────────────────────────────────\n")
        return len(missing) == 0


if __name__ == "__main__":
    # Run:  python paths.py
    # to verify all folders exist before training.
    all_ok = P.check()
    if not all_ok:
        print("Fix the missing paths above before running any training script.")
