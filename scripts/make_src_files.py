import numpy as np
import os
import shutil
import re 

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

LABELFOLDERS = ["data\\trainingdata_temporal\\labels2\\labels", "data\\folkert_miniset\\train", "data\\folkert_miniset\\test"]
OUTPUTFOLDER = "data\\src_files"
SOURCE_PATH = "data\\trainingdata_temporal\\temporal_frames\\stride2_window14"

# test_once = True
# for folder in os.listdir(SOURCE_PATH):
#     fpath = os.path.join(SOURCE_PATH, folder)
#     dest_files = os.listdir(fpath)
#     if len(dest_files)==7:
#         dest_file = sorted_nicely(dest_files)[3]
#         if test_once: print(dest_file)
#         shutil.copy(os.path.join(fpath, dest_file), os.path.join(OUTPUTFOLDER, 'images'))
#     test_once=False

for f in os.listdir(os.path.join(OUTPUTFOLDER, "images")):
    # format0 = f.replace("jpeg", "txt")
    # path0 = os.path.join(LABELFOLDERS[0], format0)
    # if os.path.exists(path0):
    #     shutil.copy(path0, os.path.join(OUTPUTFOLDER, 'labels'))
    #     continue

    # datetime, vid_id, ext = f.split("_")
    # format1 = f"{datetime}_{vid_id}.MP4_{ext.replace('jpeg', 'txt')}"
    # path1 = os.path.join(LABELFOLDERS[1], format1)
    # if os.path.exists(path1):
    #     shutil.copy(path1, os.path.join(OUTPUTFOLDER, 'labels', f.replace('jpeg', 'txt')))
    #     continue

    # datetime, vid_id, ext = f.split("_")
    # format2 = f"{datetime}_{vid_id}.MP4_{ext.replace('jpeg', 'txt')}"
    # path2 = os.path.join(LABELFOLDERS[2], format2)
    # if os.path.exists(path2):
    #     shutil.copy(path2, os.path.join(OUTPUTFOLDER, 'labels', f.replace('jpeg', 'txt')))
    #     continue
    f2 = f.replace('jpeg', 'txt')
    if not os.path.exists(os.path.join(OUTPUTFOLDER, 'labels', f2)):
        print(f2)

    