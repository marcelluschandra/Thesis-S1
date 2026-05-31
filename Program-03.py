% Program Training Model
import os, json, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import segmentation_models_pytorch as smp

import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_IMAGES = r"..."
TRAIN_LABELS = r"..."
TEST_IMAGES  = r"..."
TEST_LABELS  = r"..."

LABEL_NAME_TO_ID = {
    "background": 0,
    "encrusting": 1,
    "plate": 2,
    "massive": 3,
    "folios": 4,
    "branching": 5,
}

MINORITY_CLASSES = [4, 5]

NUM_CLASSES = 6
IGNORE_INDEX = 255

NUM_EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-4
INPUT_SIZE = (512, 512)

def normalize_label(label):
    return label.lower().strip().replace(" ", "").replace("-", "")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

class LabelMeDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def load_mask(self, json_path, size):
        with open(json_path) as f:
            data = json.load(f)

        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)

        for shp in data["shapes"]:
            raw_label = shp["label"]
            label = normalize_label(raw_label)

            if label not in LABEL_NAME_TO_ID:
                continue

            cls = LABEL_NAME_TO_ID[label]

            pts = shp["points"]
            draw.polygon([tuple(p) for p in pts], fill=cls)

        return mask

    def has_minority(self, mask):
        arr = np.array(mask)
        return any((arr == c).any() for c in MINORITY_CLASSES)

    def augment_fn(self, img, mask):
        if random.random() < 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)

        if random.random() < 0.3:
            img = TF.vflip(img); mask = TF.vflip(mask)

        if self.has_minority(mask):
            if random.random() < 0.5:
                angle = random.randint(-30, 30)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

            if random.random() < 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

        return img, mask

    def __getitem__(self, idx):
        img_path, json_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = self.load_mask(json_path, img.size)

        img = img.resize(INPUT_SIZE[::-1], resample=Image.BILINEAR)
        mask = mask.resize(INPUT_SIZE[::-1], resample=Image.NEAREST)

        if self.augment:
            img, mask = self.augment_fn(img, mask)

        img = TF.to_tensor(img)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

def get_pairs(img_dir, label_dir):
    pairs = []
    for img in Path(img_dir).glob("*"):
        json_path = Path(label_dir) / f"{img.stem}.json"
        if json_path.exists():
            pairs.append((img, json_path))
    return pairs


def compute_metrics(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)

    mask = labels != IGNORE_INDEX
    preds = preds[mask]
    labels = labels[mask]

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t, p in zip(labels, preds):
        confusion[t.long(), p.long()] += 1

    accuracy = torch.diag(confusion).sum().float() / confusion.sum().float()

    iou_per_class = []
    for i in range(num_classes):
        TP = confusion[i, i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP

        denom = TP + FP + FN
        iou = TP.float() / denom.float() if denom != 0 else torch.tensor(0.0)
        iou_per_class.append(iou)

    miou = torch.mean(torch.stack(iou_per_class))

    return accuracy.item(), miou.item(), confusion


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAME_TO_ID.keys(),
        yticklabels=LABEL_NAME_TO_ID.keys()
    )

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Best Confusion Matrix")

    plt.savefig("best_confusion_matrix.png")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = get_pairs(TRAIN_IMAGES, TRAIN_LABELS)
    val_pairs   = get_pairs(TEST_IMAGES, TEST_LABELS)

    train_ds = LabelMeDataset(train_pairs, augment=True)
    val_ds   = LabelMeDataset(val_pairs, augment=False)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, BATCH_SIZE)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        classes=NUM_CLASSES
    ).to(device)

    weights = torch.tensor([1.0,1.0,1.0,1.0,2.0,2.0]).to(device)

    criterion = FocalLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_miou = 0
    patience = 10
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs).argmax(1)

                all_preds.append(preds.cpu())
                all_labels.append(masks.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc, miou, confusion = compute_metrics(all_preds, all_labels, NUM_CLASSES)

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Train Loss     : {avg_loss:.4f}")
        print(f"Nilai Accuracy : {acc:.4f}")
        print(f"Nilai mIoU     : {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            no_improve = 0

            torch.save(model.state_dict(), "deeplabv3plus_resnet101-07.pt")

            plot_confusion_matrix(confusion.numpy())

            print("Best model saved")

        else:
            no_improve += 1

    print("\nTraining selesai")

if __name__ == "__main__":
    main()
