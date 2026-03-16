import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader

import pickle
import nibabel as nib
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PATHS
# =====================================

DATA_DIR = r"C:\Users\HP\ML_project\data"

FER_PATH = os.path.join(DATA_DIR,"FER2013")
DEAP_PATH = os.path.join(DATA_DIR,"deap-dataset","data_preprocessed_python")
FMRI_PATH = os.path.join(DATA_DIR,"fmri")

MODEL_PATH = "emotion_model.pth"

# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

# =====================================
# FER2013 DATA LOADER
# =====================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor()
])

train_data = ImageFolder(
    os.path.join(FER_PATH,"train"),
    transform=transform
)

test_data = ImageFolder(
    os.path.join(FER_PATH,"test"),
    transform=transform
)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32)

print("Emotion classes:",train_data.classes)

# =====================================
# FACE MODEL
# =====================================

class FaceModel(nn.Module):

    def __init__(self):

        super().__init__()

        base = mobilenet_v2(weights="IMAGENET1K_V1")

        base.features[0][0] = nn.Conv2d(
            1,32,3,2,1,bias=False
        )

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):

        x = self.features(x)
        x = self.pool(x)

        return x.view(x.size(0),-1)

face_model = FaceModel().to(device)

classifier = nn.Linear(1280,7).to(device)

# =====================================
# EEG FEATURE EXTRACTION (DEAP)
# =====================================

def extract_eeg_features():

    features = []
    labels = []

    files = os.listdir(DEAP_PATH)

    for file in files:

        path = os.path.join(DEAP_PATH,file)

        with open(path,'rb') as f:
            data = pickle.load(f,encoding='latin1')

        eeg = data["data"][:,:32,:]
        lab = data["labels"]

        for trial in range(eeg.shape[0]):

            trial_data = eeg[trial]

            psd_features = []

            for ch in range(32):

                f,p = welch(trial_data[ch],fs=128)

                psd_features.append(np.mean(p))

            features.append(psd_features)
            labels.append(lab[trial][0])

    return np.array(features),np.array(labels)

print("Extracting EEG features...")
eeg_X,eeg_y = extract_eeg_features()

# =====================================
# FMRI FEATURE EXTRACTION
# =====================================

def extract_fmri_features():

    features = []

    for sub in os.listdir(FMRI_PATH):

        sub_path = os.path.join(FMRI_PATH,sub)

        if not os.path.isdir(sub_path):
            continue

        for file in os.listdir(sub_path):

            if file.endswith(".nii"):

                img = nib.load(os.path.join(sub_path,file))

                data = img.get_fdata()

                mean_activation = np.mean(data)

                features.append(mean_activation)

    return np.array(features)

print("Extracting fMRI features...")
fmri_features = extract_fmri_features()

# =====================================
# TRAINING
# =====================================

if os.path.exists(MODEL_PATH):

    print("Loading saved model...")

    checkpoint = torch.load(MODEL_PATH)

    face_model.load_state_dict(checkpoint["face"])
    classifier.load_state_dict(checkpoint["classifier"])

else:

    print("Training FER model...")

    optimizer = torch.optim.Adam(
        list(face_model.parameters()) + list(classifier.parameters()),
        lr=0.0003
    )

    criterion = nn.CrossEntropyLoss()

    epochs = 3

    for epoch in range(epochs):

        face_model.train()

        total_loss = 0

        for imgs,labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            features = face_model(imgs)

            logits = classifier(features)

            loss = criterion(logits,labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss:{total_loss:.3f}")

    torch.save({
        "face":face_model.state_dict(),
        "classifier":classifier.state_dict()
    },MODEL_PATH)

# =====================================
# EVALUATION
# =====================================

face_model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for imgs,labels in test_loader:

        imgs = imgs.to(device)

        features = face_model(imgs)

        logits = classifier(features)

        preds = torch.argmax(logits,1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true,y_pred,zero_division=0))

# =====================================
# CONFUSION MATRIX
# =====================================

emotion_labels = train_data.classes

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels
)

plt.title("Emotion Confusion Matrix",fontsize=16)
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()