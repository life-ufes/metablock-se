## Metablock-SE: PAD-UFES-20 experiments

Minimal guide to preprocess metadata and run k-fold experiments combining images with tabular or sentence-embedded metadata.

## Setup

- Python 3.10+
- Install dependencies (PyTorch per your CUDA/CPU setup):

```bash
python -m pip install -r raug/requirements.txt
python -m pip install sentence-transformers model2vec timm
# Optional local package (enables `raug.*` imports)
python -m pip install -e ./raug
```

Update `config.py` paths to match your environment:

- `PAD_20_PATH` points to PAD-UFES-20 root (expects `images/` and `metadata.csv`).
- `PAD_20_IMAGES_FOLDER` points to the images folder under that root.
- `DATA_PATH` is where generated CSVs will be written (default: `data/pad-ufes-20`).

Example expected layout:

```
/datasets/PAD-UFES-20/
	images/
	metadata.csv
```

## 1) Preprocess metadata (k-fold splits + missingness)

Run both preprocessors to generate one-hot encoded and sentence-based CSVs with 5-fold patient-grouped splits. These read `config.PAD_20_RAW_METADATA` and write CSVs into `config.DATA_PATH`.

- One-hot encoded CSVs:

```bash
python -m benchmarks.pad20.preprocess.onehot
```

- Sentence CSVs (anamnese strings for sentence-transformers):

```bash
python -m benchmarks.pad20.preprocess.sentence
```

Outputs (filenames):

- `pad-ufes-20-one-hot-missing-{0|5|10|20|30|50|70}.csv`
- `pad-ufes-20-sentence-missing-{0|5|10|20|30|50|70}.csv`

## 2) Run k-fold experiments

Launch the experiment driver. It will iterate models, missingness levels, and folds configured inside `benchmarks/kfoldexperiment.py` and log runs with Sacred into `benchmarks/pad20/results/...`.

```bash
python -m benchmarks.kfoldexperiment
```

Notes:

- In `benchmarks/kfoldexperiment.py`, adjust:
	- `feature_fusion_methods`: use `metablock` for one-hot; use `metablock-se` for sentence-embeddings; use `None` for image-only.
	- `models` and `missing_percentages` as desired.
- The `folder_experiment` automatically picks the correct metadata file based on `_preprocessing` and will encode sentences with `_llm_type` (default `sentence-transformers/paraphrase-albert-small-v2`).

## Tips

- Ensure the dataset paths in `config.py` are valid before running.
- If you use GPUs, install a matching PyTorch build from pytorch.org.
- Results, configs, and metrics are stored per-fold in `benchmarks/pad20/results/...` via Sacred observers.

## 3) Aggregate predictions and plot 📊

After the k-fold runs finish, aggregate per-fold metrics into a single CSV and then plot performance vs. missing metadata rate.

### Expected results folder structure

Under `benchmarks/pad20/results/opt_<optimizer>_early_stop_<metric>/<TIMESTAMP>/` the structure is:

```
benchmarks/pad20/results/
	opt_adam_early_stop_loss/
		<TIMESTAMP>/                # e.g., 17551255817303946
			no_metadata/
				<model>/
					missing_0/
						folder_1/predictions_best_test.csv
						folder_2/predictions_best_test.csv
						... folder_5/
			metablock/
				<model>/
					missing_{0|5|10|20|30|50|70}/
						folder_{1..5}/predictions_best_test.csv
			metablock-se/
				<model>/
					missing_{0|5|10|20|30|50|70}/
						folder_{1..5}/predictions_best_test.csv
```

Tip: ensure you have the image-only baseline (`no_metadata/missing_0`) for each model; the plotting script uses it as a reference band.

### 3.1 Aggregate metrics into a CSV

This scans the timestamped results folder, collects `predictions_best_test.csv` across folds, and writes `agg_metrics.csv` at the timestamp root.

```bash
python -m utils.aggpredictions --timestamp <TIMESTAMP>
```

Output: `benchmarks/pad20/results/opt_adam_early_stop_loss/<TIMESTAMP>/agg_metrics.csv` with a multi-index:

- index: fusion in {no_metadata, metablock, metablock-se}, model, missing in {0,5,10,20,30,50,70}, metric in {balanced_accuracy, auc, f1_score}
- columns: AVG, STD, FOLDER-1..FOLDER-5

### 3.2 Plot metric vs. missing metadata

This reads `agg_metrics.csv` and produces a figure at repo root.

```bash
python -m utils.plotmissing --timestamp <TIMESTAMP> --metric balanced_accuracy
```

Where `--metric` is one of: `balanced_accuracy`, `auc`, `f1_score`.

Output: `./<metric>_vs_missing_data.png` (e.g., `balanced_accuracy_vs_missing_data.png`).

