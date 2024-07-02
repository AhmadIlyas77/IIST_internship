import pandas as pd
import cv2
import random
import os

# Load the CSV file
train_df = pd.read_csv(
    "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/training_set_pixel_size_and_HC.csv")

# Extract filenames from the dataframe
X_train_images = train_df['filename'].values

# Generate 100 random file numbers
file_no = []
for i in range(100):
    x = random.randint(0, 999)
    file_no.append(f"{x:03d}")  # Ensure the number is zero-padded to three digits

# Define base paths
base_image_path = "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/training_set/training_set/"
base_mask_path = "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/Masks/"
val_image_path = "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/val/"
val_mask_path = "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/val_mask/"

# Ensure output directories exist
os.makedirs(val_image_path, exist_ok=True)
os.makedirs(val_mask_path, exist_ok=True)

# Process images and masks
for numb in file_no:
    image_path = f"{base_image_path}{numb}_HC.png"
    mask_path = f"{base_mask_path}{numb}_HC_Annotation.png"

    # Log file paths
    print(f"Processing image: {image_path}")
    print(f"Processing mask: {mask_path}")

    # Read the image
    image = cv2.imread(image_path, 0)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # Write the image to the validation directory
    output_image_path = f"{val_image_path}{numb}_HC.png"
    cv2.imwrite(output_image_path, image)

    # Read the mask
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Error: Unable to load mask at {mask_path}")
        continue

    # Write the mask to the validation directory
    output_mask_path = f"{val_mask_path}{numb}_HC_Annotation.png"
    cv2.imwrite(output_mask_path, mask)
