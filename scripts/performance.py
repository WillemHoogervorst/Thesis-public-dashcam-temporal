import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sn
import csv

def make_log_dir(base_dir):
    stop_cond = True
    i = 0
    while stop_cond:
        i += 1
        if(not os.path.exists(f'{base_dir}/run{i}')):
            os.mkdir(f'{base_dir}/run{i}')
            stop_cond = False
            return f'run{i}'

def PR_plot(y, preds, network, run, nc=4, class_labels=['Give way', 'Speed limit', 'Keep Left/Right', 'Traffic light']):
    y = y.type(torch.int64)
    y_true = F.one_hot(y, num_classes=nc)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(nc):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], preds[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], preds[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), preds.ravel()
    )
    average_precision["micro"] = average_precision_score(y_true, preds, average="micro")

    _, ax = plt.subplots(1, 1, figsize=(9, 6))

    for i in range(nc):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"{class_labels[i]}")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="all classes", linewidth=3, color='blue')

    handles, labels = display.ax_.get_legend_handles_labels()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_title("Precision-Recall curve of " + run)
    plt.tight_layout()

    save_dir = os.path.join(f'results_{network}', run)
    if os.path.exists(save_dir):
        plt.savefig(os.path.join(save_dir, 'PR_curve.png'))
    else:
        raise Exception(f'PR curve could not be saved: path {save_dir} does not exist')

def F1_score(true, preds, class_labels=['Give way', 'Speed limit', 'Keep Left/Right', 'Traffic light']):
    true = np.array(true)
    preds = torch.argmax(preds, dim=1)
    preds = np.array(preds)
    f1_micro = f1_score(true, preds, average='micro')
    f1_macro = f1_score(true, preds, average='macro')
    return (f1_micro, f1_macro)

def Conf_Matrix_plot(conf_mat, network, run, class_labels=['Give way', 'Speed limit', 'Keep Left/Right', 'Traffic light']):
    fig = plt.figure(figsize=(12,9), tight_layout=True)
    array = np.transpose(conf_mat)
    array = array / (array.sum(0).reshape(1,-1) + 1E-9)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    sn.heatmap(
        array, 
        annot=True, 
        cmap='Blues', 
        fmt='.2f',
        square=True, 
        vmin=0.0,
        xticklabels=class_labels, 
        yticklabels=class_labels)

    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    save_dir = os.path.join(f'results_{network}', run)
    if os.path.exists(save_dir):
        fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    else:
        raise Exception(f'Confusion matrix could not be saved: path {save_dir} does not exist')

def Loss_plot(path, network, run_name):
    epochs = []
    train_loss = []
    val_loss = []
    accuracy = []
    with open(path, 'r', encoding='UTF8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            accuracy.append(float(row['accuracy']))
            
    sn.set()
    fig, axs = plt.subplots(2,2, figsize=(8,10), sharey='row')
    axs = axs.ravel()
    axs[0].plot(epochs, train_loss, marker='.', linewidth=2, markersize=10)
    axs[1].plot(epochs, val_loss, marker='.', linewidth=2, markersize=10)
    axs[2].plot(epochs, accuracy, marker='.', linewidth=2, markersize=10)
    axs[0].set_title('Train loss', fontsize=15)
    axs[1].set_title('Validation loss', fontsize=15)
    axs[2].set_title('Validation Accuracy', fontsize=15)
    axs[0].set_ylabel('Loss')
    axs[2].set_ylabel('Accuracy (%)')
    axs[3].set_axis_off()
    for ax in axs:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    fig.savefig(f'results_{network}/{run_name}/Loss.png')
