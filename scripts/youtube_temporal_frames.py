import cv2
import os
import shutil
import glob
import torch
import numpy as np


MINIBATCH_STRIDE = 1
MINIBATCH_LENGTH = 7
ROI = 1.5

VIDEOFILES_FOLDER = ["data/youtube_experiment/raw_vid"]
OUTPUTFOLDER = "data/youtube_experiment/temporal_frames_stride1"
LABELFOLDER = "data/youtube_experiment/single_labels_sample1"


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
        if f.endswith("txt"):
            video_filename = 'Youtube_dashcam.mp4'
            framenum = int(f.split("_")[-1].split("-")[0])
            vid_path = VIDEOFILES_FOLDERS[0] + "/" + video_filename

            with open(LABELFOLDER + "/" + f) as label_file:
                _, x, y, w, h = np.loadtxt(label_file, delimiter=" ") # list([class, x, y, w, h])
            
            cap = cv2.VideoCapture(vid_path)

            new_dir = OUTPUTFOLDER + "/" + f.split('.')[0]
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            else:
                continue

            framenums = range(framenum - (WINDOW_LENGTH // 2), framenum + (WINDOW_LENGTH // 2) + 1, STRIDE)

            for j in framenums:
                cap.set(1, j)
                retval, frame = cap.read()
                if retval:
                    img_h, img_w, _ = frame.shape
                    xyxy = xywh2xyxy([x, y, w*ROI, h*ROI], w=img_w, h=img_h) # list([x1, y1, x2, y2])
                    sub_img = frame[make_int(xyxy[1]):make_int(xyxy[3]), make_int(xyxy[0]):make_int(xyxy[2])]
                    resized_img = cv2.resize(sub_img, sub_img_dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(
                        f"{new_dir}/{j}.jpeg",
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

    expected_foldersize = len(range(100 - (WINDOW_LENGTH // 2), 100 + (WINDOW_LENGTH // 2) + 1, STRIDE))
    for dir in os.listdir(OUTPUTFOLDER):
        if(len(os.listdir(OUTPUTFOLDER + "/" + dir)) != expected_foldersize):
            # shutil.rmtree(OUTPUTFOLDER + "/" + dir)
            print(f"{dir} contains missing frames")
            labels = glob.glob(f'{LABELFOLDER}/{dir}*.txt')
            # for f in labels:
            #     os.remove(f)
            print(f'there are {len(labels)} corresponding labels')


if __name__ == '__main__':
    process_videos(VIDEOFILES_FOLDER, OUTPUTFOLDER, LABELFOLDER, MINIBATCH_STRIDE, MINIBATCH_LENGTH, ROI)