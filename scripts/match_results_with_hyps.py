import csv
import os
import yaml

csv_path = 'results_4D/f1_score.csv'

yaml.add_multi_constructor('', lambda loader, suffix, node: None)
yaml.add_multi_constructor('!', lambda loader, suffix, node: None)
yaml.add_multi_constructor('tag:', lambda loader, suffix, node: None, Loader=yaml.SafeLoader)


header = ['run_name', 'f1_micro', 'f1_macro', 'accuracy', 'minibatch_stride', 'minibatch_length', 'ROI']
with open('results_4D/hypsearch.csv', 'w', encoding='UTF8', newline='') as new_f:
    writer = csv.writer(new_f)
    writer.writerow(header)

with open(csv_path, 'r', encoding='UTF8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_name = row['run_name']
        with open(os.path.join('results_4D', run_name, 'hyps.yaml')) as y:
            hyps = yaml.load(y, Loader = yaml.SafeLoader)
            hyp_subset = {key: value for key, value in hyps.items() if key in ['minibatch_stride', 'minibatch_length', 'ROI']}
            row.update(hyp_subset)
            with open('results_4D/hypsearch.csv', 'a', encoding='UTF8', newline='') as new_f:
                writer = csv.DictWriter(new_f, fieldnames=header)
                writer.writerow(row)

