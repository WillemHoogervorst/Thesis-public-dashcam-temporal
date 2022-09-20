import cv2
import os
import shutil
import glob
import torch
import numpy as np


# STRIDE = 2
# WINDOW_LENGTH = 14

# VIDEOFILES_FOLDERS = ["data/trainingdata_temporal/raw_vids", "data/trainingdata_temporal/raw_vids2"]
OUTPUTFOLDER = "data/youtube_experiment/frame_sample1"
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

def process_video(OUTPUTFOLDER):
    missing_frames_cnt = 0
    if not os.path.exists(OUTPUTFOLDER):
        os.makedirs(OUTPUTFOLDER)
    video_filedir = 'data/youtube_experiment/raw_vid/Youtube_dashcam.mp4'

    cap = cv2.VideoCapture(video_filedir)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)

    # framenums = range(1, frame_count, 200)
    # framenums = [11290]

    # for j in framenums:
    #     cap.set(1, j)
    #     retval, frame = cap.read()
    #     if retval:
    #         cv2.imwrite(
    #             f"{OUTPUTFOLDER}/'Youtube_dashcam'_{j}.jpeg",
    #             frame,
    #         )
    #     else:
    #         missing_frames_cnt += 1
    #         continue

    print("# frames out of bounds: ", missing_frames_cnt)

process_video(OUTPUTFOLDER)