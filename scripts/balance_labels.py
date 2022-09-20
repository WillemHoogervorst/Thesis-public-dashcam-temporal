import os
import shutil
import numpy as np
from numpy import genfromtxt


OUTPUTFOLDER = "data\\trainingdata_temporal\\labels_balanced"
SOURCE_PATHS = ["data\\trainingdata_temporal\\labels", "data\\trainingdata_temporal\\labels3"]
CLASSES = [0,1,2,3]
MAX_COUNT = 111

def update_dict(dict: dict, key, value=1):
    if key in dict.keys():
        dict[key]+=value

os.makedirs(OUTPUTFOLDER, exist_ok=True)
copied_dict = dict([(k,0) for k in CLASSES])

for i, path in enumerate(SOURCE_PATHS):
    for f in os.listdir(path):
        annotations = genfromtxt(os.path.join(path, f), delimiter=' ')
        label = annotations[0]
        if copied_dict[label] < MAX_COUNT:
            datetime, vid_id, framenum = f.split('_')
            new_fname = f'{datetime}_{vid_id}.MP4_{framenum}' if i==1 else f
            shutil.copyfile(os.path.join(path, f), os.path.join(OUTPUTFOLDER, new_fname))
            update_dict(copied_dict, label)