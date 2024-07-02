import os
import cv2
import shutil
import random
import pandas as pd
from modules import img_crop, ellip_fill, img_augumentation

# Define data directories
data_folder = '../data/'
HC18_training_set_folder = "C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/training_set/training_set"

# Clear '../data/' folder except 'HC18_dataset'
for f in os.listdir(data_folder):
    f_path = os.path.join(data_folder, f)
    if f != 'HC18_dataset':
        if os.path.isdir(f_path):
            shutil.rmtree(f_path)
        else:
            os.remove(f_path)

# Create directories for preprocessed images and labels
train_img_folder = os.path.join(data_folder, 'train/images/')
train_label_folder = os.path.join(data_folder, 'train/labels/')
os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)

# Process images in the original training set
for i, img_name in enumerate(os.listdir(HC18_training_set_folder)):
    print(f"Dividing images and labels: {i+1} / {len(os.listdir(HC18_training_set_folder))}")
    img_path = os.path.join(HC18_training_set_folder, img_name)
    img = cv2.imread(img_path, 0)

    # Crop image
    crop_img = img_crop(img)

    # Save cropped image
    save_path = os.path.join(train_label_folder if img_name.endswith('Annotation.png') else train_img_folder, img_name)
    cv2.imwrite(save_path, crop_img)

# Fill ellipse to create segmentation mask for model training
ellip_fill(train_label_folder, train_label_folder)

# Split data into training and validation sets with a ratio of 8:2
val_img_folder = os.path.join(data_folder, 'validation/images/')
val_label_folder = os.path.join(data_folder, 'validation/labels/')
os.makedirs(val_img_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

train_images = os.listdir(train_img_folder)
random.shuffle(train_images)
val_size = round(0.2 * len(train_images))

for i, img_name in enumerate(train_images[:val_size]):
    print(f"Dividing the train and val set: {i+1} / {val_size}")
    label_name = img_name.replace('.png', '_Annotation.png')

    # Move image and label to validation set
    shutil.move(os.path.join(train_img_folder, img_name), os.path.join(val_img_folder, img_name))
    shutil.move(os.path.join(train_label_folder, label_name), os.path.join(val_label_folder, label_name))

# Create CSV file for validation set with pixel size and head circumference
val_pixel_size_hc_save = os.path.join(data_folder, 'validation/val_set_pixel_size_and_HC.csv')
p_size_hc = pd.read_csv('C:/Users/786ah/PycharmProjects/torch_test/CSM/csv/training_set_pixel_size_and_HC.csv')

val_filenames = [os.path.basename(f) for f in os.listdir(val_img_folder)]
val_psize_hc = p_size_hc[p_size_hc['filename'].isin(val_filenames)]
val_psize_hc.to_csv(val_pixel_size_hc_save, index=False)

# Data augmentation in training set
train_augu_img_folder = os.path.join(data_folder, 'train/augu_images/')
train_augu_label_folder = os.path.join(data_folder, 'train/augu_labels/')
os.makedirs(train_augu_img_folder, exist_ok=True)
os.makedirs(train_augu_label_folder, exist_ok=True)

# Image augmentation
img_augumentation(train_img_folder, train_augu_img_folder, data="image")

# Label augmentation
img_augumentation(train_label_folder, train_augu_label_folder, data="label")

print("Data preprocessing finished!")
