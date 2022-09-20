import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


def xywh2xyxy(x, w=160, h=160, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = np.rint(w * (x[0] - x[2] / 2) + padw)  # top left x
    y[1] = np.rint(h * (x[1] - x[3] / 2) + padh)  # top left y
    y[2] = np.rint(w * (x[0] + x[2] / 2) + padw)  # bottom right x
    y[3] = np.rint(h * (x[1] + x[3] / 2) + padh)  # bottom right y
    return y

def make_int(x):
    return max(int(x), 0)

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
        img_set_dir = self.main_dir + "/" + label_filename.split(".")[0]
        image_files = os.listdir(img_set_dir)
        
        with open(self.labels_dir + "/" + label_filename) as f:
            label, _, _, _, _ = np.loadtxt(f, delimiter=" ") # list([class, x, y, w, h])

        def load_img(f):
            img = cv2.imread(img_set_dir + "/" + f, cv2.IMREAD_UNCHANGED)
            return(self.transform(img))

        tensor_image_stack = torch.stack([load_img(f) for f in image_files], dim=1)

        return tensor_image_stack, label