import os
import shutil
from numpy import genfromtxt
import numpy

# VIDEOFILES = os.listdir("../../data/trainingdata_temporal")
OUTPUTFOLDER = "data\\trainingdata_temporal\\labels"
SOURCE_PATHS = ["data\\train", "data\\test"]
ORIGINAL_CLASSES = [0,1,4,5,8]
CLASS_MAP = {0:0, 1:1, 4:2, 5:2, 8:3} # {k:v} = original:new labels
FORMAT = ['%i', '%.9f', '%.9f', '%.9f', '%.9f']

os.makedirs(OUTPUTFOLDER, exist_ok=True)
for path in SOURCE_PATHS:
    for f in os.listdir(path):
        if f.startswith("202") and f.endswith(".txt"):
            annotations = genfromtxt(os.path.join(path, f), delimiter=' ')
            annotations = [annotations] if len(annotations.shape)==1 else annotations
            idx=0
            for a in annotations:
                if(len(a)>0 and a[0] in CLASS_MAP.keys()):
                    a[0] = CLASS_MAP[a[0]]
                    base, ext = os.path.splitext(f)
                    numpy.savetxt(os.path.join(OUTPUTFOLDER, f'{base}-{idx}{ext}'), [a], delimiter=' ', fmt=FORMAT)
                    idx += 1