import os
import pandas as pd
import numpy as np
from PIL import Image

# =====================================================
# CHANGE THIS PATH TO YOUR DATASET FOLDER
# =====================================================

DATASET_PATH = r"C:\Users\HP\ML_project\data\MNE-sample-data"

print("\n===== DATASET ROOT CONTENT =====\n")
for item in os.listdir(DATASET_PATH):
    print(item)

# =====================================================
# SHOW FULL FOLDER STRUCTURE
# =====================================================

print("\n===== FULL DIRECTORY STRUCTURE =====\n")

for root, dirs, files in os.walk(DATASET_PATH):
    
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 4 * level
    
    print(f"{indent}{os.path.basename(root)}/")
    
    subindent = ' ' * 4 * (level + 1)
    
    for f in files:
        print(f"{subindent}{f}")

# =====================================================
# CHECK CSV FILE STRUCTURE
# =====================================================

print("\n===== CSV FILE ANALYSIS =====\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".csv"):

            path = os.path.join(root,file)

            print("\nCSV FILE:",path)

            try:
                df = pd.read_csv(path,nrows=5)

                print("\nColumns:")
                print(df.columns)

                print("\nSample rows:")
                print(df.head())

            except Exception as e:
                print("Could not read:",e)

# =====================================================
# CHECK NUMPY FILES
# =====================================================

print("\n===== NUMPY FILE ANALYSIS =====\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".npy"):

            path = os.path.join(root,file)

            data = np.load(path)

            print("\nFile:",file)
            print("Shape:",data.shape)
            print("Datatype:",data.dtype)

# =====================================================
# CHECK IMAGE FILES
# =====================================================

print("\n===== IMAGE FILE ANALYSIS =====\n")

image_extensions = (".png",".jpg",".jpeg",".bmp")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith(image_extensions):

            path = os.path.join(root,file)

            try:
                img = Image.open(path)

                print("\nImage:",file)
                print("Size:",img.size)
                print("Mode:",img.mode)

                break

            except:
                pass