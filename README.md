Hello! Before using this program, there are a few things you should know.

This program was developed to perform coral-reef recognition and coverage-area calculation using the DeepLabV3+ model with a ResNet-101 backbone.
It also utilizes orthomosaic images to reconstruct the structure and area of the coral reef.
Agisoft Metashape Professional is used to support the 3D reconstruction process.

If you want to use this program, please follow the guidelines below:
1. The Dataset folder contains coral-reef images collected from northern Taiwan.
2. There are 720 images, divided into two sets: 70% for training and 30% for testing.
3. All images include polygon-based annotations stored as JSON files (located inside the labels folder).
4. Inside Program.ipynb, there are three main components:
   - Training Program
   - Detection Program
   - Coverage Calculation Program
5. Run the programs in this exact order:
   Training → Detection → Coverage Calculation.
6. Make sure all repository paths are correctly set to avoid errors related to missing or incorrect file locations.
7. The coverage-area calculation results will be exported as a CSV file.

Thank you for using this program. If you wish to further develop or improve it, your contributions are very welcome.
