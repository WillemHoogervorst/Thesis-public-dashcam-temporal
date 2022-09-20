import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


class CustomDataSet(Dataset):
    def __init__(self, data_dir, labels_dir, transform, sub_img_dim=(160,160)):
        self.main_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.sub_img_dim = sub_img_dim

        self.all_img_dirs = os.listdir(data_dir)
        self.all_labels = os.listdir(labels_dir)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        label_filename = self.all_labels[idx]
        img_dir = self.main_dir + "/" + label_filename.replace(".txt", ".jpeg")
        
        with open(self.labels_dir + "/" + label_filename) as f:
            label, _, _, _, _ = np.loadtxt(f, delimiter=" ") # list([class, x, y, w, h])

        def load_img(f):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            return(self.transform(img))

        tensor_image = load_img(img_dir)

        return tensor_image, label