import os
import numpy as np
from numpy import genfromtxt
import cv2
import csv
import torch
from statistics import mode
import shutil

### TO DO: ###
# Make most frequent classes dynamic: replace static 1 & 6

filedir = "data/folkert_miniset"

def update_dict(dict: dict, key, value=1):
    if key in dict.keys():
        dict[key]+=value

def count_classes(classes, data_path, use_mode=True):
    class_count = dict([(k,0) for k in classes])
    for f in os.listdir(data_path):
        if f.endswith("txt"):
            with open(data_path + '/' + f, "r") as fp:
                lines = fp.readlines()
                ys=[int(c.split(" ")[0]) for c in lines]
                if use_mode:
                    update_dict(class_count, mode(ys))
                else:
                    for y in ys:
                        update_dict(class_count, y)

    return class_count

def extract_classlabels(imageset, classes):
    filenames=[]

    def add_filename(f, c):
        if os.path.exists(filedir + f"/{imageset}/" + f.replace('txt', 'png')):
            filenames.append([f.replace('txt', 'png'), c])
        elif os.path.exists(filedir + f"/{imageset}/" + f.replace('txt', 'jpeg')):
            filenames.append([f.replace('txt', 'jpeg'), c])

    for f in os.listdir(filedir + f"/{imageset}/"):
        if f.endswith("txt"):
            with open(filedir + f"/{imageset}/" + f, "r") as fp:
                lines = fp.readlines()
                ys=[int(c.split(" ")[0]) for c in lines]
                most_freq = mode(ys)
                if most_freq < len(classes):
                    if ys.count(6)>0:
                        add_filename(f, 6) # slightly balance the data towards class 6: ANWB
                    else:
                        add_filename(f, most_freq)

    return filenames

def resize_images(filenames, dim, imageset):
    classes = []
    for filename, cl in filenames:
        img = cv2.imread(f'{filedir}/{imageset}/{filename}', cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(f'data/folkert_miniset_resized/{imageset}/{filename}', resized)
        classes.append(cl)
    return classes

def xywhn2xyxy(x, w=160, h=160, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = np.rint(w * (x[:, 0] - x[:, 2] / 2) + padw)  # top left x
    y[:, 1] = np.rint(h * (x[:, 1] - x[:, 3] / 2) + padh)  # top left y
    y[:, 2] = np.rint(w * (x[:, 0] + x[:, 2] / 2) + padw)  # bottom right x
    y[:, 3] = np.rint(h * (x[:, 1] + x[:, 3] / 2) + padh)  # bottom right y
    return y

def snap_from_image_bounding_box(imageset, dim):
    def make_int(x):
        return max(int(x), 0)
    classes = []
    set_dir = f"{filedir}/{imageset}"
    create_dir = f'data/folkert_miniset_snap_resized/{imageset}/'
    os.makedirs(create_dir, exist_ok=True)
    for f in os.listdir(set_dir):
        if f.endswith("txt"):
            # load corresponding image:
            im_dir = f"{set_dir}/{f}"
            extension = ''
            if os.path.exists(f"{set_dir}/{f.replace('txt', 'png')}"):
                extension = 'png'
            elif os.path.exists(f"{set_dir}/{f.replace('txt', 'jpeg')}"):
                extension = 'jpeg'
            im_dir = im_dir.replace('txt', extension)
            img = cv2.imread(im_dir, cv2.IMREAD_UNCHANGED)
            if img is None: # check if image is present
                print('Wrong path:', im_dir)
            else:
                img_h, img_w, _ = img.shape
                # read bounding box specs from txt file:
                with open(f"{set_dir}/{f}", "r") as fp:
                    bounding_boxes = np.loadtxt(fp, delimiter=" ") # list([class, x, y, w, h])
                    if len(bounding_boxes.shape) == 1:
                        bounding_boxes = bounding_boxes[:, None].T
                    bbs = bounding_boxes[:, 1:5] # list([x, y, w, h])
                    xyxy = xywhn2xyxy(bbs, w=img_w, h=img_h) # list([x1, y1, x2, y2])
                # crop, resize and save subimage within bounding box:
                for index, coords in enumerate(xyxy): 
                    new_img_name = f.replace('.txt', f'{index}.{extension}')
                    sub_img = img[make_int(coords[1]):make_int(coords[3]), make_int(coords[0]):make_int(coords[2])]
                    resized = cv2.resize(sub_img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(create_dir + new_img_name, resized)
                    classes.append(int(bounding_boxes[index, 0]))
    print(f"Done extracting {imageset} bbox snaps.")
    return classes

def create_subset(classes, source_dir, new_dir, imageset, countdict: dict, balance=False):
    def label_mapper(l):
        if l in [0,1]: # give way & speed limit
            return l
        if l in [4,5]: # keep left/right
            return 2
        if l == 8: # stoplicht
            return 3
    
    label_path = f"{source_dir}/labels/{imageset}_classes.csv"
    source_labels = genfromtxt(label_path, delimiter='\n')
    # def filter_func(c):
    #     return c in classes
    # filtered_labels = filter(filter_func, source_labels)
    min_dict = dict([(label_mapper(k),0) for k in classes])
    for k in classes:
        key = label_mapper(k)
        update_dict(min_dict, key, countdict[k])
    min_count = min(min_dict.values())
    copied_dict = dict([(label_mapper(k),0) for k in classes])
    print(copied_dict, countdict, min_dict, min_count)
    dest_labels = []
    os.makedirs(f'{new_dir}/{imageset}/', exist_ok=True)
    for i, f in enumerate(os.listdir(f"{source_dir}/{imageset}")):
        label = int(source_labels[i])
        if label in classes and copied_dict[label_mapper(label)] < min_count:
            dest_labels.append(label_mapper(label))
            shutil.copyfile(f"{source_dir}/{imageset}/{f}", f"{new_dir}/{imageset}/{f}")
            update_dict(copied_dict, label_mapper(label))
    os.makedirs(f'{new_dir}/labels/', exist_ok=True)
    with open(f'{new_dir}/labels/{imageset}_classes.csv', 'w+', newline='') as csv_1:
        csv_out = csv.writer(csv_1)
        csv_out.writerows([[l] for l in dest_labels])
    print(f"copied {len(dest_labels)} {imageset} files to {new_dir}/{imageset}")



    

if __name__ == '__main__':
    # classes = range(0,9) # All classes
    classes = [0,1,2,3] # Target classes
    countdict = count_classes(classes, 'data/trainingdata_temporal/labels_balanced', False) # 1-class: 93, multi-class: 233
    # countdict2 = count_classes(classes, 'data/trainingdata_temporal/labels3', False) # 1-class: 93, multi-class: 233
    # sortdict = dict(sorted(countdict.items(), reverse=True, key=lambda item: item[1]))
    # print(countdict, countdict2, sum(countdict.values()) + sum(countdict2.values())) # 307, 111, 55, 249 -> 307, 111, 118, 249
    print(countdict, sum(countdict.values())) # 

    # for imageset in ['train', 'test']:
    #     create_subset(classes, 'data/folkert_miniset_snap_resized', 'data/folkert_miniset_snap_resized_subset_balanced', imageset, countdict, balance=True)
    #     continue
    #     filenames = extract_classlabels(imageset, classes) # ['namestring.png', class]
    #     # Y = resize_images(filenames, (160,160), imageset)
    #     Y = snap_from_image_bounding_box(imageset, (160,160))
    #     mode_Y = mode(Y)
    #     print(f'Most common {imageset} class: {mode_Y} with {Y.count(mode_Y)}/{len(Y)} occurrences')
    #     # with open(f'data/folkert_miniset_resized/labels/{imageset}_classes.csv', 'w', newline='') as csv_1:
    #     os.makedirs(f'data/folkert_miniset_snap_resized/labels/', exist_ok=True)
    #     with open(f'data/folkert_miniset_snap_resized/labels/{imageset}_classes.csv', 'w+', newline='') as csv_1:
    #         csv_out = csv.writer(csv_1)
    #         csv_out.writerows([Y[index]] for index in range(0, len(Y)))
    #     print(f"# extracted {imageset} files: {len(filenames)}")
