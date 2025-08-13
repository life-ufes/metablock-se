import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from utils.simulatemissingdata import simulate_missing_data
import config
from benchmarks.pad20.dataset import PAD20
from functools import partial

def label_encode_non_nans(df, label_encoder, feature):
    was_na = df[feature].isna()
    df[feature] = label_encoder.fit_transform(df[feature])
    df.loc[was_na, feature] = np.nan

def preprocess(df_folder, folder_path):
    cnn_results = pd.read_csv(folder_path / 'metadata_with_preds.csv')
    df_folder['img_id'] = df_folder['img_id'].str.replace('.png', '')

    df_folder = df_folder.merge(cnn_results[['img_id', 'folder'] + [f'diagnostic_cnn_{l}' for l in PAD20.LABELS]],
                                on='img_id', how='left')
    
    df_folder = df_folder.rename(columns={f'diagnostic_cnn_{l}':l for l in PAD20.LABELS})

    # encode targets
    df_folder['diagnostic_number'] = label_encoder.fit_transform(df_folder['diagnostic'])

    # we use the same folders as in the vision baseline cross-validation to avoid data leakeage
    train_mask = df_folder['folder'] != folder_number

    # discretize numerical features
    df_folder['age_group'] = pd.cut(df_folder['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    df_folder['diameter'] = pd.cut(df_folder['diameter'], bins=[0, 5, 10, 15, 20, 25, 30, 35, 120], include_lowest=True)

    # encode categorical features as numbers (keeping nan values)
    for feature in ['age_group', 'diameter', 'region']:
        label_encode_non_nans(df_folder, label_encoder, feature)

    # save csv
    df_folder['split'] = 'train'
    df_folder['split'] = df_folder['split'].where(train_mask, other='val')
    return df_folder[['img_id', 'split', 'diagnostic', 'diagnostic_number', 'itch', 'grew', 'hurt', 
                        'changed', 'bleed', 'elevation', 'age_group', 'diameter', 'region'] + PAD20.LABELS]

def save_csv(df, missing, raw, save_folder, folder_number):
    file_name = f'folder_{folder_number}_raw' if raw else f'folder_{folder_number}'
    file_name += f'_missing_{missing}.csv'
    df.to_csv(save_folder / file_name, index=False)

if __name__ == '__main__':
    try:
        cnn_only_results_folder = next(Path('.').rglob(str(config.IMAGE_ONLY_EXPERIMENT)))
    except StopIteration:
        raise FileNotFoundError(f'No folder found for experiment {config.IMAGE_ONLY_EXPERIMENT}.')

    df = pd.read_csv(config.PAD_20_RAW_METADATA)
    df = df.replace('UNK', np.nan)

    # group diameter as max(diameter_1, diameter_2)
    df['diameter'] = df.apply(lambda row: max(row['diameter_1'], row['diameter_2']), axis=1)

    # fix booleans
    df = df.replace(['True', 'False'], [1, 0])

    label_encoder = LabelEncoder()

    os.makedirs(config.DATA_PATH, exist_ok=True)

    for model_path in (p for p in cnn_only_results_folder.iterdir() if p.is_dir()):
        model_name = model_path.stem
        for folder_number in range(1, 6):
            folder_path = model_path / f'folder_{folder_number}'
            if not folder_path.exists():
                raise FileNotFoundError(f'Folder {folder_path} does not exist.')

            df_folder = df.copy()
            os.makedirs(config.DATA_PATH / model_name, exist_ok=True)
            simulate_missing_data(df_folder, save_folder=config.DATA_PATH,
                            save_raw_metadata_with_missing_values = True,
                            encoder_function=partial(preprocess, folder_path=folder_path),
                            encoder_name= 'bayesian',
                            features=PAD20.RAW_CATEGORICAL_FEATURES + PAD20.NUMERICAL_FEATURES,
                            dataset='pad-ufes-20', save_function=partial(save_csv, save_folder=config.DATA_PATH / model_name, folder_number=folder_number)),
