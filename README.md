# Introduction 
Code used in the Thesis written by Willem Hoogervorst, between 01/04/2022 - 29/09/2022.
Contains two Convolutional Neural Networks: a 3D CNN and a 4D CNN.
Used to compare the performance for models with and without using temporal information fusion.

# Getting Started
1.	Installation process
```
git clone
```
2.	Software dependencies
```
pip install -r requirements.txt
```
3.  Data
```
Google 'Yolo annotation' to find out how to annotate the data before preprocessing.
Preprocessing for the 4D CNN is done using a script: ~/scripts/make_temporal_set.py
During preprocessing, images become 160 x 160 px. Labels should be a single line with a class label and xywh bbox coordinates.
```

4.  Run training
```
First, specify the paths and hyperparameters in run_train_3D.py or run_train_4D.py
# starts training, LOGS stdout, returns process ID:
nohup python run_train_[3D|4D].py > LOGS/[LOGNAME].txt &
# kill process ID with: 
kill -9 [PID]
tail -f LOGS/[LOGNAME].txt  # watch logs
watch -n1 nvidia-smi        # watch gpu
htop                        # watch cpu
column -s, -t < loss.csv | less -#2 -N -S # read loss.csv from results
# results are collected in a folder under results_[3D|4D]/run[x]
```

5.  Run inference
```
First, specify the paths and hyperparameters in run_inf_[3D|4D].py
The model_path should lead to a trained .pth file
# starts inference:
python run_inf_[3D|4D].py
# results are collected in a folder under results_[3D|4D]/run[x]
```
