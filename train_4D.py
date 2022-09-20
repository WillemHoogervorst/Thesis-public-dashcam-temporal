# library imports
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
# from torchsummary import summary
from torchinfo import summary
import numpy as np
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import sklearn.metrics as metrics
import csv
import shutil
import time
import yaml

# local imports
from models.network_4D import NeuralNetwork
from models.dataset_4D import CustomDataSet
from models.utils import EarlyStopping
from scripts.performance import *
from scripts.make_temporal_set import process_videos

torch.cuda.empty_cache()

def typecast(tensor, cputype, cudatype, device):
        if device == 'cpu':
            return tensor.type(cputype)
        else:
            return tensor.type(cudatype)

def train(dataloader, model, loss_fn, optimizer, run_name, device):
    size = len(dataloader.dataset)
    model.train()
    current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X).to(device)
        # print(pred[:10])
        loss = loss_fn(pred, typecast(y, torch.LongTensor, torch.cuda.LongTensor, device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 2 == 0:
        loss = loss.item()
        if (batch * len(y) < current):
            current = size  
            # print(f"  loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")
        else: 
            current = (batch + 1) * len(y)
            # print(f"  loss: {loss:>7f}  [{current:>4d}/{size:>4d}]", end='\r')
    return loss


def test(dataloader, model, loss_fn, last_epoch, run_name, device, nc=4):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    raw_preds = torch.empty((0,nc),device=device)
    all_y = torch.empty(0, device=device)
    conf_mat = np.zeros((4,4))

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            np_pred = torch.argmax(pred, dim=1)
            test_loss += loss_fn(pred, typecast(y, torch.LongTensor, torch.cuda.LongTensor, device)).item()
            correct += typecast((np_pred == y), torch.FloatTensor, torch.cuda.FloatTensor, device).sum().item()
            if last_epoch:
                all_y = torch.cat((all_y, typecast(y, torch.LongTensor, torch.cuda.LongTensor, device)), dim=0)
                raw_preds = torch.cat((raw_preds, pred), dim=0)
                batch_conf_mat = metrics.confusion_matrix(y.cpu(), np_pred.cpu(), labels=[0,1,2,3], sample_weight=np.full(y.shape, 1/size))
                conf_mat += batch_conf_mat


    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    # print(f"  Test Acc: {(accuracy):>0.1f}%")
    # Rows = true labels, cols = pred labels. E.g. first row, second column is true label 0 predicted as label 1.
    if last_epoch:
        PR_plot(all_y.cpu(), raw_preds.cpu(), network='4D', run=run_name, nc=nc)
        Conf_Matrix_plot(conf_mat, network='4D', run=run_name)
        f1_micro, f1_macro = F1_score(all_y.cpu(), raw_preds.cpu())
        with open(os.path.join('results_4D', run_name, 'preds.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(raw_preds.tolist())
        with open(os.path.join('results_4D', run_name, 'true.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([y] for y in all_y.tolist())
        with open(os.path.join('results_4D', 'youtube_experiment', 'f1_score.csv'), 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_name, f1_micro, f1_macro, accuracy])

    return (test_loss, accuracy)            



def run(hyps, paths, t0, change_data=True):
    # Set cpu or gpu device for training.
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    # print('Using device', device)

    lr_low = hyps["lr_low"] # = 1e-4
    lr_high = hyps["lr_high"] # = 1e-3
    train_test_split = hyps["train_test_split"] # = 0.7
    batch_size = hyps["batch_size"] # = 64
    network_size = hyps["network_size"] # = 32: amount of channels created from RGB input in first convolution
    wh = hyps["wh"] # = 160: Width and Height of input images (px)
    loss_fn = hyps["loss_fn"] # = nn.NLLLoss()
    optimizer = hyps["optimizer"]
    epochs = hyps["epochs"]
    patience = hyps["patience"]
    conv_threshold = hyps["conv_threshold"]
    minibatch_stride = hyps["minibatch_stride"]
    minibatch_length = hyps["minibatch_length"] # Input depth: Number of frames in each window
    ROI = hyps["ROI"]

    video_files_path = paths["video_files_path"] # 'path/to/blobstore?'
    output_files_path = paths["output_files_path"] # "data/temporal_frames"
    label_files_path = paths["label_files_path"] # "data/labels_balanced"

    if change_data:
        if os.path.exists(output_files_path):
            shutil.rmtree(output_files_path)
        process_videos(video_files_path, output_files_path, label_files_path, minibatch_stride, minibatch_length, ROI) 
    
    t_int = time.time()
    # print(f"Minutes elapsed after data processing: {((t_int - t0)/60):>0.2f}")

    full_dataset = CustomDataSet(output_files_path, label_files_path, transform=ToTensor())
    train_size = int(train_test_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    training_data, test_data = random_split(full_dataset, [train_size, test_size])

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = NeuralNetwork(network_size, minibatch_length, wh).to(device)
    # summary(model, input_size=(batch_size,3,minibatch_length,wh,wh)) # 3 for RGB

    optimizer_high = optimizer(model.parameters(), lr=lr_high)
    optimizer_low = optimizer(model.parameters(), lr=lr_low)

    run_name = make_log_dir('results_4D')
    f = open(f'results_4D/{run_name}/hyps.yaml', 'w+')
    yaml.dump(hyps, f, allow_unicode=True)
    stopper, stop = EarlyStopping(patience=patience, conv_threshold=conv_threshold), False

    with open(f'results_4D/{run_name}/loss.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'val_loss', 'accuracy']
        writer.writerow(header)
    print('\n\n')
    for t in range(epochs):
        # print(f"\033[F\033[F\033[FEpoch {t+1}") # direct pointer three lines back
        last_ep = t==epochs-1 or stopper.possible_stop
        optimizer = optimizer_high if t in range(3, int(epochs/3)) else optimizer_low
        train_loss = train(train_dataloader, model, loss_fn(), optimizer, run_name, device)
        val_loss, acc = test(test_dataloader, model, loss_fn(), last_ep, run_name, device)
        stop = stopper(t, train_loss, val_loss, acc)
        with open(f'results_4D/{run_name}/loss.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([t, train_loss, val_loss, acc])
        if stop:
            break

    Loss_plot(f'results_4D/{run_name}/loss.csv', network='4D', run_name=run_name)
    print('Done! Saving results in results_4D/', run_name, sep='')
    torch.save(model.state_dict(), f"results_4D/{run_name}/model.pth")

def inference(hyps, paths):
    batch_size = hyps["batch_size"] # = 64
    network_size = hyps["network_size"] # = 32: amount of channels created from RGB input in first convolution
    wh = hyps["wh"]
    loss_fn = hyps["loss_fn"] # = nn.NLLLoss()
    minibatch_length = hyps["minibatch_length"] # Input depth: Number of frames in each window

    temporal_frames_path = paths["temporal_frames_path"] # 'path/to/blobstore?'
    label_files_path = paths["label_files_path"] # "data/labels_balanced"
    model_path = paths["model_path"]

    device='cpu'
    test_data = CustomDataSet(temporal_frames_path, label_files_path, transform=ToTensor())
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    model = NeuralNetwork(network_size, minibatch_length, wh).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    run_name = make_log_dir('results_4D/youtube_experiment')

    _, _ = test(test_dataloader, model, loss_fn(), True, f"youtube_experiment\\{run_name}", device)