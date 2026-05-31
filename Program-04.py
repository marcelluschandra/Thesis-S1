% Coral Detection Program
import os
import math
import numpy as np
from PIL import Image

import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

from tqdm import tqdm

MODEL_PATH = r"C:\Workspace\TEEP\ModelTraining\Program\checkpoints\deeplabv3plus_resnet101-07.pt"

ORTHO_IMAGE = r"C:\Workspace\TEEP\ModelTraining\Program\Test\Orthomosaic.png"

OUTPUT_MASK = "pred_mask.png"
OUTPUT_COLOR = "pred_color.png"
OUTPUT_OVERLAY = "pred_overlay.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
TILE_SIZE = 512
OVERLAP = 128
INPUT_SIZE = (512, 512)
ALPHA = 0.5
USE_MORPHOLOGY = True
MORPH_KERNEL = 3

CLASS_WEIGHTS = [
    0.85,   # background
    0.75,   # encrusting
    1.15,   # plate
    1.30,   # massive
    1.35,   # folios
    1.35    # branching
]

BASE_COLORS = [
    (0, 0, 0),        # background
    (240, 0, 20),     # encrusting
    (0, 100, 255),    # plate
    (255, 255, 0),    # massive
    (138, 43, 226),   # folios
    (50, 75, 255)     # branching
]

print("Loading model...")

model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,
    classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

print("Model loaded")
print("\nLoading orthomosaic...")
ortho = Image.open(ORTHO_IMAGE).convert("RGB")
W, H = ortho.size
print(f"Image size : {W} x {H}")
step = TILE_SIZE - OVERLAP
x_positions = list(range(0, W, step))
y_positions = list(range(0, H, step))
total_tiles = len(x_positions) * len(y_positions)
print(f"Total tiles : {total_tiles}")
prob_map = np.zeros(
    (NUM_CLASSES, H, W),
    dtype=np.float32
)
weight_map = np.zeros(
    (H, W),
    dtype=np.float32
)
yy, xx = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
center_y = TILE_SIZE / 2
center_x = TILE_SIZE / 2
distance = np.sqrt(
    (xx - center_x) ** 2 +
    (yy - center_y) ** 2
)
distance = distance / distance.max()
tile_weight = 1.0 - distance
tile_weight = np.clip(tile_weight, 0.05, 1.0)
tile_weight = tile_weight.astype(np.float32)

def predict_with_tta(tensor):
    preds = []
    logits = model(tensor)
    preds.append(
        F.softmax(logits, dim=1)
    )
    flip_h = torch.flip(
        tensor,
        dims=[3]
    )
    logits_h = model(flip_h)
    logits_h = torch.flip(
        logits_h,
        dims=[3]
    )
    preds.append(
        F.softmax(logits_h, dim=1)
    )
    flip_v = torch.flip(
        tensor,
        dims=[2]
    )
    logits_v = model(flip_v)

    logits_v = torch.flip(
        logits_v,
        dims=[2]
    )
    preds.append(
        F.softmax(logits_v, dim=1)
    )
    probs = torch.mean(
        torch.stack(preds),
        dim=0
    )

    return probs

print("\nRunning inference...")

with torch.no_grad():
    pbar = tqdm(total=total_tiles)
    for y in y_positions:
        for x in x_positions:
            x_end = min(x + TILE_SIZE, W)
            y_end = min(y + TILE_SIZE, H)

            tile = ortho.crop(
                (x, y, x_end, y_end)
            )
            original_w = x_end - x
            original_h = y_end - y
            tile_resized = tile.resize(
                INPUT_SIZE,
                resample=Image.BILINEAR
            )
            tensor = TF.to_tensor(
                tile_resized
            ).unsqueeze(0).to(DEVICE)
            probs = predict_with_tta(tensor)
            probs = probs.squeeze(0).cpu().numpy()
            resized_probs = []
            for c in range(NUM_CLASSES):
                prob_img = Image.fromarray(
                    probs[c]
                )
                prob_img = prob_img.resize(
                    (original_w, original_h),
                    resample=Image.BILINEAR
                )
                resized_probs.append(
                    np.array(prob_img)
                )
            probs = np.stack(resized_probs)
            local_weight = tile_weight[
                :original_h,
                :original_w
            ]
            for c in range(NUM_CLASSES):

                prob_map[
                    c,
                    y:y_end,
                    x:x_end
                ] += (
                    probs[c] * local_weight
                )
            weight_map[
                y:y_end,
                x:x_end
            ] += local_weight
            pbar.update(1)
    pbar.close()

print("\nNormalizing probability map...")

prob_map /= np.maximum(
    weight_map,
    1e-6
)

print("Applying class balancing...")

for class_id in range(NUM_CLASSES):

    prob_map[class_id] *= CLASS_WEIGHTS[class_id]

print("Generating final mask...")

final_mask = np.argmax(
    prob_map,
    axis=0
).astype(np.uint8)

if USE_MORPHOLOGY:

    print("Applying morphology refinement...")

    kernel = np.ones(
        (MORPH_KERNEL, MORPH_KERNEL),
        np.uint8
    )

    refined_mask = np.zeros_like(final_mask)

    for cls in range(NUM_CLASSES):

        binary = (
            final_mask == cls
        ).astype(np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        refined_mask[binary == 1] = cls

    final_mask = refined_mask

Image.fromarray(final_mask).save(
    OUTPUT_MASK
)
print("Mask grayscale saved")
print("Creating color mask...")
color_mask = np.zeros(
    (H, W, 3),
    dtype=np.uint8
)
for class_id, color in enumerate(BASE_COLORS):
    color_mask[
        final_mask == class_id
    ] = color
Image.fromarray(color_mask).save(
    OUTPUT_COLOR
)
print("Color mask saved")
print("Creating overlay...")
ortho_np = np.array(ortho)
overlay = (
    ortho_np * (1 - ALPHA) +
    color_mask * ALPHA
).astype(np.uint8)
Image.fromarray(overlay).save(
    OUTPUT_OVERLAY
)
print("Overlay saved")
print("Inference selesai")

print(f"Mask     : {OUTPUT_MASK}")
print(f"Color    : {OUTPUT_COLOR}")
print(f"Overlay  : {OUTPUT_OVERLAY}")
