import numpy as np
import os
import shutil
from numpy import genfromtxt


TRAIN_PROPORTION = 70
LABELFOLDER = "data\\src_files\\labels"
SOURCE_PATH = "data\\src_files\\images"
OUTPUTFOLDER = f"data\\yolo_balanced\\train{TRAIN_PROPORTION}"
CLASS_MAP = {0:0, 1:1, 4:2, 5:2, 8:3} # {k:v} = original:new labels
FORMAT = ['%i', '%.9f', '%.9f', '%.9f', '%.9f']

os.makedirs(os.path.join(OUTPUTFOLDER, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTFOLDER, 'test'), exist_ok=True)

def adjust_labels(filepath):
    annotations = genfromtxt(filepath, delimiter=' ')
    annotations = [annotations] if len(annotations.shape)==1 else annotations
    returnval = []
    for a in annotations:
        if(len(a)>0 and a[0] in CLASS_MAP.keys()):
            if len(a) == 6:
                a = a[:-1]
            a[0] = CLASS_MAP[a[0]]
            returnval.append(a)
    return returnval


test_once = True
for f in os.listdir(SOURCE_PATH):
    if np.random.random() < (TRAIN_PROPORTION / 100):
        shutil.copy(os.path.join(SOURCE_PATH, f), os.path.join(OUTPUTFOLDER, 'train'))
        label_filename = f.replace('jpeg', 'txt')
        new_content = adjust_labels(os.path.join(LABELFOLDER, label_filename))
        np.savetxt(os.path.join(OUTPUTFOLDER, 'train', label_filename), new_content, delimiter=' ', fmt=FORMAT)
    else:
        shutil.copy(os.path.join(SOURCE_PATH, f), os.path.join(OUTPUTFOLDER, 'test'))
        label_filename = f.replace('jpeg', 'txt')
        new_content = adjust_labels(os.path.join(LABELFOLDER, label_filename))
        np.savetxt(os.path.join(OUTPUTFOLDER, 'test', label_filename), new_content, delimiter=' ', fmt=FORMAT)

    test_once = False