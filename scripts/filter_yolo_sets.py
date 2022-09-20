import os
import shutil

VIDEOFILES = "data/"
OUTPUTFOLDER = "data/yolo_snap_resized_subset_balanced/"
SOURCE_PATH = "data/"

def label_mapper(l):
        if l in [0,1]: # give way & speed limit
            return str(l)
        if l in [4,5]: # keep left/right
            return '2'
        if l == 8: # stoplicht
            return '3'

for subset in ['train', 'test']:
    print("Processing " + subset)
    os.makedirs(OUTPUTFOLDER + subset, exist_ok=True)
    for f in os.listdir(VIDEOFILES + subset):
        base_filename, ext = os.path.splitext(f)
        base_filename = base_filename[:-1] # remove last char
        base_source = f"{SOURCE_PATH + subset}/{base_filename}"
        # Copy image
        if not os.path.exists(f"{OUTPUTFOLDER}{subset}/{base_filename}{ext}"):
            shutil.copy(f"{base_source}{ext}", OUTPUTFOLDER + subset)
        # Copy labels
        if not os.path.exists(f"{OUTPUTFOLDER}{subset}/{base_filename}.txt"):
            shutil.copy(f"{base_source}.txt", f"{OUTPUTFOLDER}{subset}/{base_filename}.txt")

for subset in ['train', 'test']:
    print("Relabeling " + subset)
    for f in os.listdir(OUTPUTFOLDER + subset):
        if f.endswith(".txt"):
            filedir = OUTPUTFOLDER + subset + '/' + f
            # Read
            with open(filedir, 'r+') as file:
                lines = file.readlines()
            # Adjust
            new_lines = []
            for i, line in enumerate(lines):
                new_class = label_mapper(int(line[0]))
                if(new_class):
                    new_lines.append(line.replace(line[0], new_class, 1))
            # Write
            with open(filedir, "w") as write_file:
                write_file.truncate(0)
                write_file.seek(0)
                write_file.writelines(new_lines)