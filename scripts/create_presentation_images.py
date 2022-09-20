import cv2
import os
import shutil
import glob
import torch
import numpy as np


# STRIDE = 2
# WINDOW_LENGTH = 14

BASE_DIR = "data/trainingdata_temporal"
VIDEOFILES_FOLDERS = ["raw_vids", "raw_vids2"]
# VID_NAME = "20211223154907_001682F.MP4"
# VID_NAME = "20220113145117_001801F.MP4"
VID_NAME = "20220616054100_004455F.MP4"
# FRAMENUM = 5027
# FRAMENUM = 985
FRAMENUM = 1975
OUTPUTFOLDER = "data/presentation/example_frame_original"
# INPUTFOLDER = "data/trainingdata_temporal/labels_balanced"


# FILENAME FORMATS:
# video: datetime_vidID.MP4
# label: datetime_vidID.MP4_framenum-instance.txt
# frames_dir: datetime_vidID.MP4_framenum
# frame: datetime_vidID.MP4_framenum.jpeg
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

def create_example_image(BASE_DIR, VIDEOFILES_FOLDERS, OUTPUTFOLDER, VID_NAME, FRAMENUM):
    if not os.path.exists(OUTPUTFOLDER):
        os.makedirs(OUTPUTFOLDER)
    for folder in VIDEOFILES_FOLDERS:

        if os.path.exists(os.path.join(BASE_DIR, folder, VID_NAME)):
            cap = cv2.VideoCapture(os.path.join(BASE_DIR, folder, VID_NAME))
            cap.set(1, FRAMENUM)
            retval, frame = cap.read()

            if retval:
                cv2.imwrite(
                    f"{OUTPUTFOLDER}/{VID_NAME}_{FRAMENUM}.jpeg",
                    frame,
                )

if __name__ == '__main__':
    create_example_image(BASE_DIR, VIDEOFILES_FOLDERS, OUTPUTFOLDER, VID_NAME, FRAMENUM)