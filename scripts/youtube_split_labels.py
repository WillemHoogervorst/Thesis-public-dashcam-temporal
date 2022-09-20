import os
import shutil
from numpy import genfromtxt
import numpy

# VIDEOFILES = os.listdir("../../data/trainingdata_temporal")
OUTPUTFOLDER = "data\\youtube_experiment\\single_labels_sample1"
SOURCE_PATH = "data\\youtube_experiment\\labels_sample1"
FORMAT = ['%i', '%.9f', '%.9f', '%.9f', '%.9f']

os.makedirs(OUTPUTFOLDER, exist_ok=True)
for f in os.listdir(SOURCE_PATH):
    annotations = genfromtxt(os.path.join(SOURCE_PATH, f), delimiter=' ')
    annotations = [annotations] if len(annotations.shape)==1 else annotations
    idx=0
    for a in annotations:
        if(len(a)>0):
            base, ext = os.path.splitext(f)
            numpy.savetxt(os.path.join(OUTPUTFOLDER, f'{base}-{idx}{ext}'), [a], delimiter=' ', fmt=FORMAT)
            idx += 1