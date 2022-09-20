from torch.optim import Adagrad
from torch.nn import NLLLoss
import csv
import time
import os

import train_3D

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
    "temporal_frames_path": "data/youtube_experiment/static_frames",
    "label_files_path": "data/youtube_experiment/single_labels_sample1",
    "model_path": "results_3D/run3/model.pth"
}


if __name__ == "__main__":
    os.makedirs('results_3D/youtube_experiment', exist_ok=True)
    if not os.path.exists('results_3D/youtube_experiment/f1_score.csv'):
        with open('results_3D/youtube_experiment/f1_score.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['run_name', 'f1_micro', 'f1_macro', 'accuracy']
            # write the header
            writer.writerow(header)

    EXTRA_HYPS.update(HYPS)
    train_3D.inference(EXTRA_HYPS, LOCAL_PATHS)
