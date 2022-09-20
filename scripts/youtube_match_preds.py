import torch
import torch.nn.functional as F

torch.set_printoptions(precision=2, sci_mode=False)
with open('results_4D/youtube_experiment/run4/preds.csv', "r") as fp:
    lines = fp.readlines()
    print(lines[0])
    data = [[float(x) for x in y.split(',')] for y in lines]

preds = torch.tensor(data, dtype=torch.float64)

preds_exp = torch.exp(preds)

for i in range(preds_exp.shape[0]):          
    min_ele = torch.min(preds_exp[i])
    preds_exp[i] -= min_ele
    preds_exp[i] /= torch.max(preds_exp[i])

with open('results_4D/youtube_experiment/run4/true.csv', "r") as fp:
    lines = fp.readlines()
    data = [[float(x) for x in y.split(',')] for y in lines]

true = torch.tensor(data, dtype=torch.float64)

combined = torch.cat((preds_exp, true), dim=1)

print(combined[0:5,:])