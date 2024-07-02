import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2


train_df = pd.read_csv("C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/training_set_pixel_size_and_HC.csv")

X_train_images = train_df['filename'].values

def mask(name):
    image =  cv2.imread("C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/training_set/training_set/"+name+"_Annotation.png",0)
    im = image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(im, ellipse, (255,255,255), -1)

    cv2.imwrite("C:/Users/786ah/PycharmProjects/torch_test/Fetal head circumference/Masks/"+name+"_Annotation.png", im)

for file in X_train_images:
    name = file[0:len(file) - 4]
    mask(name)
