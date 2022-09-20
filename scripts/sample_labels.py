import os
import shutil
import numpy as np
from numpy import genfromtxt


# VIDEOFILES = os.listdir("../../data/trainingdata_temporal")
OUTPUTFOLDER = "data\\trainingdata_temporal\\labels3"
SOURCE_PATH = "data\\trainingdata_temporal\\labels2\\labels"
ORIGINAL_CLASSES = [0,1,4,5,8]
CLASS_MAP = {0:0, 1:1, 4:2, 5:2, 8:3} # {k:v} = original:new labels
FORMAT = ['%i', '%.9f', '%.9f', '%.9f', '%.9f']
PROBABILITY = 0.05
MIN_FRAME_DIST = 50

def sample_labels():
    os.makedirs(OUTPUTFOLDER, exist_ok=True)
    prev_frame_num = 0
    for f in os.listdir(SOURCE_PATH):
        if np.random.random() > PROBABILITY: continue
        annotations = genfromtxt(os.path.join(SOURCE_PATH, f), delimiter=' ')
        annotations = [annotations] if len(annotations.shape)==1 else annotations
        idx=0
        for a in annotations:
            if(len(a)>0 and a[0] in CLASS_MAP.keys()):
                base, ext = os.path.splitext(f)
                # 
                curr_frame_num = int(base.split('_')[-1])
                if abs(curr_frame_num - prev_frame_num) < MIN_FRAME_DIST and curr_frame_num != prev_frame_num:
                    break
                else:
                    prev_frame_num = curr_frame_num
                a[0] = CLASS_MAP[a[0]]
                np.savetxt(os.path.join(OUTPUTFOLDER, f'{base}-{idx}{ext}'), [a[:-1]], delimiter=' ', fmt=FORMAT)
                idx += 1

def update_dict(dict: dict, key, value=1):
    if key in dict.keys():
        dict[key]+=value

def count_classes(classes, data_path):
    class_count = dict([(k,0) for k in classes])
    prev_frame_num = 0
    for f in os.listdir(data_path):
        base, ext = os.path.splitext(f)
        # 
        curr_frame_num = int(base.split('_')[-1])
        if abs(curr_frame_num - prev_frame_num) < MIN_FRAME_DIST and curr_frame_num != prev_frame_num:
            continue
        else:
            prev_frame_num = curr_frame_num
        with open(data_path + '/' + f, "r") as fp:
            lines = fp.readlines()
            ys=[int(c.split(" ")[0]) for c in lines]
            for y in ys:
                update_dict(class_count, y)

    return class_count

def add_keep_right():
    def sample_from_batch(batch):
        if(len(batch)==1):
            return batch[0]
        else:
            return np.random.choice(batch)

    prev_frame_num = 0
    counter = 0
    frame_batch = []

    for f in os.listdir(SOURCE_PATH):
        annotations = genfromtxt(os.path.join(SOURCE_PATH, f), delimiter=' ')
        annotations = [annotations] if len(annotations.shape)==1 else annotations

        for a in annotations:
            if(len(a)>0 and a[0]==5):
                base, ext = os.path.splitext(f)
                curr_frame_num = int(base.split('_')[-1])
                if len(frame_batch)==0:
                    frame_batch.append(f)
                    prev_frame_num = curr_frame_num
                elif (abs(curr_frame_num - prev_frame_num) < MIN_FRAME_DIST and curr_frame_num != prev_frame_num):
                    frame_batch.append(f)
                else:
                    the_chosen_one = sample_from_batch(frame_batch)
                    annotations2 = genfromtxt(os.path.join(SOURCE_PATH, the_chosen_one), delimiter=' ')
                    annotations2 = [annotations2] if len(annotations2.shape)==1 else annotations2
                    idx=0
                    for a2 in annotations2:
                        if(len(a2)>0 and a2[0]==5):
                            base, ext = os.path.splitext(the_chosen_one)
                            a2[0] = CLASS_MAP[a2[0]]
                            new_dir = os.path.join(OUTPUTFOLDER, f'{base}-{idx}{ext}')
                            if not os.path.exists(new_dir):
                                np.savetxt(new_dir, [a2[:-1]], delimiter=' ', fmt=FORMAT)
                                counter+=1
                            idx += 1
                    prev_frame_num = curr_frame_num
                    frame_batch = [f]
    return counter


if __name__ == '__main__':
    classes = [4,5] # Target classes
    # countdict = count_classes(classes, 'data/trainingdata_temporal/labels2/labels') # 1-class: 93, multi-class: 233
    # countdict2 = count_classes(classes, 'data/trainingdata_temporal/labels3') # 1-class: 93, multi-class: 233
    # sortdict = dict(sorted(countdict.items(), reverse=True, key=lambda item: item[1]))
    # print(countdict, countdict2, sum(countdict.values()) + sum(countdict2.values())) # 307, 101, 55, 249
    print(add_keep_right()) # 74