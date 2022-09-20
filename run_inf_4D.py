from torch.optim import Adagrad
from torch.nn import NLLLoss
import csv
import time
import os

import train_4D

HYPS = {
    "minibatch_stride": 2,
    "minibatch_length": 7,
    "ROI": 1.5
}

EXTRA_HYPS = {  
    "batch_size": 64,
    "network_size": 32, # amount of channels created from RGB input in first convolution
    "wh": 160, # Width and Height of input images (px)
    "loss_fn": NLLLoss,
    "optimizer": Adagrad
}

LOCAL_PATHS = {
    "temporal_frames_path": "data/youtube_experiment/temporal_frames_stride1",
    "label_files_path": "data/youtube_experiment/single_labels_sample1",
    "model_path": "results_4D/run11/model.pth"
}


if __name__ == "__main__":
    os.makedirs('results_4D/youtube_experiment', exist_ok=True)
    if not os.path.exists('results_4D/youtube_experiment/f1_score.csv'):
        with open('results_4D/youtube_experiment/f1_score.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['run_name', 'f1_micro', 'f1_macro', 'accuracy']
            # write the header
            writer.writerow(header)

    EXTRA_HYPS.update(HYPS)
    train_4D.inference(EXTRA_HYPS, LOCAL_PATHS)
