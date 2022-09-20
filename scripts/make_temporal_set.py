import cv2
import os
import shutil
import glob
import torch
import numpy as np


# STRIDE = 2
# WINDOW_LENGTH = 14

# VIDEOFILES_FOLDERS = ["data/trainingdata_temporal/raw_vids", "data/trainingdata_temporal/raw_vids2"]
# OUTPUTFOLDER = f"data/trainingdata_temporal/temporal_frames/stride{STRIDE}_window{WINDOW_LENGTH}"
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

def process_videos(VIDEOFILES_FOLDERS, OUTPUTFOLDER, LABELFOLDER, STRIDE, minibatch_length, ROI, sub_img_dim=(160,160)):
    missing_frames_cnt = 0
    INPUTFILES = os.listdir(LABELFOLDER)
    WINDOW_LENGTH = STRIDE * minibatch_length

    if not os.path.exists(OUTPUTFOLDER):
        os.makedirs(OUTPUTFOLDER)
    for i, f in enumerate(INPUTFILES):
        # end = '\r' if i+1<len(INPUTFILES) else '\n'
        # print(f'Extracting temporal frames: [{i+1}/{len(INPUTFILES)}]', end=end)
        if f.endswith("txt"):
            video_filename = f.split(".")[0]
            framenum = int(f.split("_")[-1].split("-")[0])
            if os.path.exists(VIDEOFILES_FOLDERS[0] + "/" + video_filename + ".MP4"):
                vid_path = VIDEOFILES_FOLDERS[0] + "/" + video_filename + ".MP4"
            elif os.path.exists(VIDEOFILES_FOLDERS[1] + "/" + video_filename + ".MP4"):
                vid_path = VIDEOFILES_FOLDERS[1] + "/" + video_filename + ".MP4"
            else:
                print(VIDEOFILES_FOLDERS[0] + "/" + video_filename + ".MP4", "or", VIDEOFILES_FOLDERS[1] + "/" + video_filename + ".MP4 not found")
                raise Exception("Videofiles not found!")


            with open(LABELFOLDER + "/" + f) as label_file:
                _, x, y, w, h = np.loadtxt(label_file, delimiter=" ") # list([class, x, y, w, h])
            
            cap = cv2.VideoCapture(vid_path)

            new_dir = OUTPUTFOLDER + "/" + f.split('-')[0]
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            else:
                continue

            framenums = range(framenum - (WINDOW_LENGTH // 2) + 1, framenum + (WINDOW_LENGTH // 2), STRIDE)

            for j in framenums:
                cap.set(1, j)
                retval, frame = cap.read()
                if retval:
                    img_h, img_w, _ = frame.shape
                    xyxy = xywh2xyxy([x, y, w*ROI, h*ROI], w=img_w, h=img_h) # list([x1, y1, x2, y2])
                    sub_img = frame[make_int(xyxy[1]):make_int(xyxy[3]), make_int(xyxy[0]):make_int(xyxy[2])]
                    resized_img = cv2.resize(sub_img, sub_img_dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(
                        f"{new_dir}/{video_filename}_{j}.jpeg",
                        resized_img,
                    )
                else:
                    missing_frames_cnt += 1
                    continue

    print("# frames out of bounds: ", missing_frames_cnt)

    # Checks:
    for f in INPUTFILES:
        check_path = OUTPUTFOLDER + "/" + f.split('-')[0]
        if not os.path.exists(check_path):
            print(f'path for {f.split("-")[0]} does not exist')
        break

    expected_foldersize = len(range(100 - (WINDOW_LENGTH // 2) + 1, 100 + (WINDOW_LENGTH // 2), STRIDE))
    for dir in os.listdir(OUTPUTFOLDER):
        if(len(os.listdir(OUTPUTFOLDER + "/" + dir)) != expected_foldersize):
            shutil.rmtree(OUTPUTFOLDER + "/" + dir)
            print(f"removed {dir} due to missing frames")
            labels = glob.glob(f'{LABELFOLDER}/{dir}*.txt')
            for f in labels:
                os.remove(f)
            print(f'removed {len(labels)} corresponding labels')