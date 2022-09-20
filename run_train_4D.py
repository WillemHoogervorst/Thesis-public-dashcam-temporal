from torch.optim import Adagrad
from torch.nn import NLLLoss
import csv
import time
import os

import train_4D

HYPS = {
    "minibatch_stride": [2, 5, 10, 20, 30],
    "minibatch_length": [5, 7, 9, 11],
    "ROI": [1.2, 1.5, 1.8, 2.1]
}

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
    "video_files_path": ["data/trainingdata_temporal/raw_vids"],
    "output_files_path": "data/trainingdata_temporal/temporal_frames/testrun7",
    "label_files_path": "data/trainingdata_temporal/labels"
}

TEST_RUN = False

if __name__ == "__main__":
    t0_0 = time.time()

    os.makedirs('results', exist_ok=True)
    with open('results_4D/f1_score.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['run_name', 'f1_micro', 'f1_macro', 'accuracy']
        # write the header
        writer.writerow(header)

    if TEST_RUN:
        EXTRA_HYPS.update({"minibatch_stride": 10, "minibatch_length":11, 'ROI': 1.5})
        PATHS["output_files_path"] = "data/temporal_frames/testrun7"
        train_4D.run(EXTRA_HYPS, PATHS, time.time(), True)
    else:
        for stride in HYPS["minibatch_stride"]:
            for window_length in HYPS["minibatch_length"]:
                for roi in HYPS["ROI"]:
                    print(f'\nStarting run: s={stride} l={window_length} r={roi}')
                    t0 = time.time()
                    EXTRA_HYPS['minibatch_stride'] = stride
                    EXTRA_HYPS['minibatch_length'] = window_length
                    EXTRA_HYPS['ROI'] = roi
                    # LOCAL_PATHS["output_files_path"] = f"data/trainingdata_temporal/temporal_frames/testrun{window_length}"
                    train_4D.run(EXTRA_HYPS, PATHS, t0, True)
                    t1 = time.time()
                    t_tot = t1-t0_0
                    h = t_tot // 3600
                    m = (t_tot / 3600 - h) * 60
                    print(f"Time elapsed so far: {int(h)}:{m:>0.2f}")

    t_total = time.time()-t0_0
    h_tot = t_total // 3600
    m_tot = (t_total / 3600 - h_tot) * 60
    print(f"Total time elapsed: {int(h_tot)}:{m_tot:>0.2f}")
    

