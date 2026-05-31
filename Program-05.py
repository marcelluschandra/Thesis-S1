% Program 05
% Coverage Calculation Program
import csv
from pathlib import Path
import numpy as np
from PIL import Image

MASK_PATH = r"C:\Workspace\TEEP\ModelTraining\Program\Test\gt_mask.png"
CSV_PATH = r"C:\Workspace\TEEP\ModelTraining\Program\coral_area_m2.csv"

PIXEL_SIZE_M = 0.00166

CLASS_NAMES = [
    "background",
    "encrusting",
    "plate",
    "massive",
    "folios",
    "branching"
]

def main():
    print("Loading mask")
    mask = np.array(Image.open(MASK_PATH))
    coral_mask = mask != 0
    total_coral_pixels = np.sum(coral_mask)
    pixel_area_m2 = PIXEL_SIZE_M ** 2
    results = []
    print("\nCORAL AREA RESULT")
    for class_id in range(1, len(CLASS_NAMES)):

        class_name = CLASS_NAMES[class_id]
        class_pixels = np.sum(mask == class_id)
        class_area_m2 = (
            class_pixels * pixel_area_m2
        )
        coverage_percent = (
            (class_pixels / total_coral_pixels) * 100
            if total_coral_pixels > 0 else 0
        )
        results.append([
            class_name,
            int(class_pixels),
            round(class_area_m2, 6),
            round(coverage_percent, 3)
        ])
        print(
            f"{class_name:<12} | "
            f"Pixels: {class_pixels:<10} | "
            f"Area: {class_area_m2:.6f} m² | "
            f"Coverage: {coverage_percent:.3f}%"
        )
    total_area_m2 = (
        total_coral_pixels * pixel_area_m2
    )
    print(f"Total Coral Pixels : {total_coral_pixels}")
    print(f"Total Coral Area   : {total_area_m2:.6f} m²")
    print("Total Coverage     : 100.000%")

    Path(CSV_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Class",
            "Pixels",
            "Area_m2",
            "Coverage_Percent"
        ])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow([
            "TOTAL_CORAL",
            int(total_coral_pixels),
            round(total_area_m2, 6),
            100.000
        ])
    print(f"\nCSV saved : {CSV_PATH}")

if __name__ == "__main__":
    main()
