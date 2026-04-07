import torch
print(torch.cuda.is_available())       # should print True
print(torch.cuda.get_device_name(0))   # e.g. "NVIDIA RTX 3060"