from pathlib import Path
import pandas as pd
from itertools import product
from metrics.metrics import get_metrics
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Aggregate metrics from a kfold validation experiment')
parser.add_argument('--timestamp', type=str, )
args = parser.parse_args()

matches = list(Path(f'./benchmarks/pad20/results/').rglob(args.timestamp))
if not matches:
    raise ValueError(f'No results were found for experiment {args.timestamp}')

base_path = matches[0]

comb_paths = [p for p in base_path.iterdir() if p.is_dir()]
def get_subdirs(path):
    return (p for p in path.iterdir() if p.is_dir())

models = ['mobilenet', 'resnet-50', 'caformer_s18', 'efficientnet-b4']
missing = [0, 5, 10, 20, 30, 50, 70]

comb_methods = ['no_metadata', 'metablock', 'metablock-se',]
metric_names = ['balanced_accuracy', 'auc', 'f1_score']

index = pd.MultiIndex.from_tuples(list(product(comb_methods, models, missing, metric_names)))
df = pd.DataFrame(columns=['AVG', 'STD'] + [f'FOLDER-{i}' for i in range(1,6)], index=index)
labels = None
folder_pattern = r'folder_(\d)'

def add_metrics(df, base_path, comb_path, model_path, missing_percentage, labels):
    metrics = {}
    for preds_path in base_path.rglob('**/predictions_best_test.csv'):
        folder_number = re.search(folder_pattern, str(preds_path)).group(0).replace('folder_', '')
        df_preds = pd.read_csv(preds_path)
        labels = labels if labels is not None else df_preds['REAL'].unique()
        metrics[folder_number] = get_metrics(df_preds['REAL'], df_preds[labels])
    
    for metric in metric_names:
        row = {}
        for folder_number, values in metrics.items():
            row[f'FOLDER-{folder_number}'] = values[metric]
        values = [v[metric] for v in metrics.values()]
        row['AVG'] = np.mean(values)
        row['STD'] = np.std(values)

        df.loc[pd.IndexSlice[comb_path.stem, model_path.stem, missing_percentage, metric]] = row

for comb_path in comb_paths:
    if comb_path.stem not in comb_methods:
        continue
    #number_id_path = next(get_subdirs(comb_path))
    #print(number_id_path)
    for model_path in get_subdirs(comb_path):
        if model_path.stem not in models:
            continue
        for missing_path in get_subdirs(model_path):
            missing_percentage = int(missing_path.stem.replace('missing_', ''))
            if missing_percentage not in missing:
                continue
            add_metrics(df, missing_path, comb_path, model_path, missing_percentage, labels)

df.index = df.index.set_names(['fusion', 'model', 'missing', 'metric'])
df.to_csv(base_path / 'agg_metrics.csv')