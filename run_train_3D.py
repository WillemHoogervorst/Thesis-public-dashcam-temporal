from torch.optim import Adagrad
from torch.nn import NLLLoss
import csv
import time
import os

import train_3D

EXTRA_HYPS = {
    "lr_low": 1e-4,
    "lr_high": 1e-3,
    "train_test_split": 0.7,    
    "batch_size": 64,
    "network_size": 32, # amount of channels created from RGB input in first convolution
    "wh": 160, # Width and Height of input images (px)
    "loss_fn": NLLLoss,
    "optimizer": Adagrad,
    "epochs": 300,
    "patience": 30,
    "conv_threshold": 1e-5
}

PATHS = {
    "video_files_path": ["../blobstore2","../blobstore2/processed"],
    "output_files_path": "data/temporal_frames/tmp",
    "label_files_path": "data/labels_balanced"
}

LOCAL_PATHS = {
    "video_files_path": ["data/trainingdata_temporal/raw_vids", "data/trainingdata_temporal/raw_vids2"],
    "output_files_path": "data/trainingdata/frames",
    # "label_files_path": "data/testing/labels"
    "label_files_path": "data/trainingdata_temporal/labels_balanced"

}

TEST_RUN = False

if __name__ == "__main__":
    t0_0 = time.time()

    os.makedirs('results', exist_ok=True)
    with open('results_3D/f1_score.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['run_name', 'f1_micro', 'f1_macro', 'accuracy']
        # write the header
        writer.writerow(header)

    if TEST_RUN:
        PATHS["output_files_path"] = "data/temporal_frames/testrun7"
        train_3D.run(EXTRA_HYPS, PATHS, True)
    else:
        train_3D.run(EXTRA_HYPS, LOCAL_PATHS, False)


    t_total = time.time()-t0_0
    h_tot = t_total // 3600
    m_tot = (t_total / 3600 - h_tot) * 60
    print(f"Total time elapsed: {int(h_tot)}:{m_tot:>0.2f}")
    

