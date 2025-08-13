import pandas as pd
from pathlib import Path
import argparse
from benchmarks.pad20.dataset import PAD20
from metrics.metrics import get_metrics, aggregate_metrics, plot_confusion_matrix
parser = argparse.ArgumentParser(description='Aggregate metrics from a kfold validation experiment')
parser.add_argument('--experiment', type=int, help='Experiment ID', required=True)
parser.add_argument('--missing', default=0, type=int, required=False)
args = parser.parse_args()

EXPERIMENT_ID = args.experiment
result = pd.DataFrame()

benchmarks = ['pad20']
comb_methods = ['metablock', 'metablock-se', 'no_metadata', 
                'cross-attention', 'bayesiannetwork',]

models = []
dataset = None
comb_method = None
save_path = None
for benchmark in benchmarks:
    dataset_path = Path(f'./benchmarks/{benchmark}/results')

    if not dataset_path.is_dir():
        continue

    for subpath in (f for f in dataset_path.iterdir() if f.is_dir()):
        matches = list(subpath.rglob(str(EXPERIMENT_ID)))
        for match in matches:
            dataset = benchmark
            comb_method = match.parent.name
            save_path = match.parent / str(EXPERIMENT_ID)
            if comb_method in comb_methods or comb_method.split('-')[0] in comb_methods and match.is_dir():
                models = [d.name for d in match.iterdir() if d.is_dir()]
                break
        if models:
            break

if not models:
    raise ValueError(f'No results were found for experiment {EXPERIMENT_ID}')

for model in models:
    path = save_path / f'{model}' / f'missing_{args.missing}'
    metrics = []
    preds = []
    targets = []
    try:
        for cp in path.rglob('**/predictions_best_test.csv'):
            csv = pd.read_csv(cp)
            metrics.append(get_metrics(csv['REAL'], csv[PAD20.LABELS]))
            plot_confusion_matrix(csv['REAL'], csv['PRED'], PAD20.LABELS, cp.parent)
            preds.extend(csv['PRED'])
            targets.extend(csv['REAL'])
        result[f'{comb_method} + {model}'] = aggregate_metrics(metrics,)
        plot_confusion_matrix(targets, preds, PAD20.LABELS, path)

    except Exception as e:
        print(f"Error processing {model} - {e}")

results = result.sort_values(by='balanced_accuracy', axis=1, ascending=False)
result.to_csv(save_path / f'agg_metrics_missing_{args.missing}.csv')
print(result)