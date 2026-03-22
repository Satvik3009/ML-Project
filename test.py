import torch
ckpt = torch.load(r"C:\Users\HP\ML_project\emotion_model_v3.pth", map_location="cpu", weights_only=False)

print("classifier.0.weight shape:", ckpt["classifier.0.weight"].shape)  # [256, 1376]

# Find all encoder output sizes
for prefix in ["face_enc", "eeg_enc", "fmri_enc"]:
    keys = [k for k in ckpt if k.startswith(prefix)]
    if keys:
        last_key = [k for k in keys if "weight" in k][-1]
        print(f"{prefix} last weight key: {last_key} → shape: {ckpt[last_key].shape}")