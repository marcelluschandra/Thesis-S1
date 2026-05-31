% Program Dataset Seperation
import os
import random
import shutil

dataset_dir = r"..." 
output_dir = r"..." 

splits = ["train", "test"]
for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

images = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

random.shuffle(images)

total = len(images)
train_size = int(0.7 * total)
test_size = total - train_size

train_files = images[:train_size]
test_files = images[train_size:]

def copy_files(file_list, dest_folder):
    for file in file_list:
        src = os.path.join(dataset_dir, file)
        dst = os.path.join(output_dir, dest_folder, file)
        shutil.copy(src, dst)

copy_files(train_files, "train")
copy_files(test_files, "test")

print(f"Total images: {total}")
print(f"Train: {len(train_files)}")
print(f"Test:  {len(test_files)}")
