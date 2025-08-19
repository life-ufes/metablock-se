import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

parser = argparse.ArgumentParser(description='Aggregate metrics from a kfold validation experiment')
parser.add_argument('--timestamp', type=str, required=True)
parser.add_argument('--metric', type=str, default='balanced_accuracy', help='[balanced_accuracy, auc, f1_score]')
args = parser.parse_args()


missing = [0, 5, 10, 20, 30, 50, 70]
comb_methods = ['metablock', 'metablock-se']
models = ['caformer_s18', 'resnet-50', 'efficientnet-b4', 'mobilenet', ]

metric_names = ['balanced_accuracy', 'auc', 'f1_score']

metric = args.metric

metric_label = {
    'weighted avg f1-score': 'F1 Score',
    'precision': 'Precision',
    'weighted avg recall': 'Recall',
    'acc': 'Accuracy',
    'loss': 'Loss',
    'balanced_accuracy': 'Balanced Accuracy',
    'auc': 'Area Under Curve',
    'f1_score': 'F1 Score',
}

def plot(df, axis, metric, df_no_metadata, model=None, show_values=False, palette=None):
    df_plot = pd.concat([df[f'FOLDER-{i}'] for i in range(1,6)])
    df_no_metadata = pd.concat([df_no_metadata[f'FOLDER-{i}'] for i in range(1,6)])
    if model:
        df_plot = df_plot.xs((model, metric), level=(1,3))
    else:
        df_plot = df_plot.xs(metric, level=3)
    df_no_metadata = df_no_metadata.xs(metric, level=3)
    # Remove the multiindex because sns does not support it
    df_plot = df_plot.reset_index()

    df_no_metadata = df_no_metadata.reset_index()
    if not model:
        df_plot.columns = ['method', 'model', 'missing', metric]
    else :
        df_plot.columns = ['method', 'missing', metric]
    df_no_metadata.columns = ['method', 'model', 'missing', metric]

    new_names = {
        'resnet-50': 'ResNet-50',
        'mobilenet': 'MobileNet-V2',
        'efficientnet-b4': 'EfficientNet-B4',
        'caformer_s18': 'CAFormer-S18',
        'metablock': 'MetaBlock',
        'metablock-se': 'MetaBlock-SE (ours)',
    }
    df_plot.replace(new_names, inplace=True)

    full_model_name = []
    for fusion_method, cnn in zip(df_plot['method'], df_plot['model'] if not model else [model] * len(df_plot)):
        fusion_method = fusion_method.replace(' (ours)', '')
        full_model_name.append(f'{fusion_method} with {cnn}{" (ours)" if fusion_method == "MetaBlock-SE" else ""}')
    df_plot['model'] = full_model_name

    unique_methods = tuple(df_plot['method'].unique())

    base_colors = sns.color_palette("tab10", 4)  # Choose 4 distinct colors

    if not model:
        for comb_method, color in zip(unique_methods, base_colors):
            if '-SE' in comb_method:
                mask = df_plot['model'].str.startswith(comb_method.replace(' (ours)', ''))
            else:
                mask = df_plot['model'].str.startswith(comb_method) & ~df_plot['model'].str.contains('-SE')
            ax = sns.lineplot(data=df_plot[mask], x='missing', y=metric, hue='model', style='model',
                ax=axis, palette=sns.light_palette(color, n_colors=10)[6:])
    else:
        ax = sns.lineplot(data=df_plot, x='missing', y=metric, hue='model', ax=axis, palette=palette)

    if show_values:
        # Add data points to figure
        first_line_y = ax.lines[1].get_ydata()
        for i, line in enumerate(ax.lines):
            for j, (x, y) in enumerate(zip(line.get_xdata(), line.get_ydata())):
                diff = first_line_y[j] - y
                offset = 0.03 if i == 0 and abs(diff) < 0.025 else 0

                offset = offset if diff < 0 else -offset
                ax.text(
                    max(x, 2.3),
                    y + offset,
                    f'{y:.2f}',
                    ha='right',
                    va='bottom',
                    fontsize=12
                )

    def plot_baseline(vision_model, linestyle='--'):
        mask = df_no_metadata['model'] == vision_model
        sample_mean = np.mean(df_no_metadata[mask][metric])
        sem = stats.sem(df_no_metadata[mask][metric])
        # Define confidence level (e.g., 95%)
        confidence_level = 0.95

        # Calculate degrees of freedom
        degrees_freedom = len(df_no_metadata[mask][metric]) - 1

        # Calculate the confidence interval
        confidence_interval = stats.t.interval(
            confidence_level,
            degrees_freedom,
            loc=sample_mean,
            scale=sem
        )

        line_value = sample_mean  # The y-value for the horizontal line
        ci_lower = confidence_interval[0]   # Lower bound of confidence interval
        ci_upper = confidence_interval[1]   # Upper bound of confidence interval

        # Add a semi-transparent confidence interval band
        axis.axhspan(ci_lower, ci_upper, alpha=0.15, color='gray')

        # Add the horizontal line
        axis.axhline(y=line_value, color='k', linestyle=linestyle, label=f'{new_names[vision_model]} baseline')

    if model:
        plot_baseline(model, linestyle=':')

    ax.set_ylabel(metric_label[metric])
    ax.set_xlabel('Missing metadata rate (%)')
    ax.set_xticks(missing)
    ax.set_yticks(np.arange(0.25, 0.8, 0.1))
    ax.set_xlim(0, max(missing))
    ax.set_ylim(0.25, 0.80)
    ax.grid(True)
    # Increase axis label and tick label font sizes
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, title_fontsize=16)


matches = list(Path(f'./benchmarks/pad20/results/').rglob(args.timestamp))
if not matches:
    raise ValueError(f'No results were found for experiment {args.timestamp}')

base_path = matches[0]

df = pd.read_csv(base_path / 'agg_metrics.csv', index_col=[0, 1, 2, 3])
df = df.drop(columns=['AVG', 'STD'])
df_no_metadata = df.loc[pd.IndexSlice['no_metadata', models, [0], metric_names]]
df = df.loc[pd.IndexSlice[comb_methods, models, missing, metric_names]]
fig, axes = plt.subplot_mosaic([['all', model] for model in models], figsize=(24, 12))

if len(models) > 1:
    for model in models:    
        plot(df, axes[model], metric, df_no_metadata, model=model, show_values=True)
plot(df, axes['all'], metric, df_no_metadata)

plt.tight_layout()

plt.savefig(f'./{metric}_vs_missing_data.png', dpi=300)