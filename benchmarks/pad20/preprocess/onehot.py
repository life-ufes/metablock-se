import config
import numpy as np
import pandas as pd

from benchmarks.pad20.dataset import PAD20
from sklearn.model_selection import StratifiedGroupKFold
from raug.raug.utils.loader import label_categorical_to_number
from utils.simulatemissingdata import simulate_missing_data

def one_hot_encode(df):
    for col in df.select_dtypes(include='category').columns:
        if 'EMPTY' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('EMPTY')
    df = df.replace(" ", np.nan).replace("  ", np.nan)
    df = df.fillna('EMPTY')
    df[PAD20.NUMERICAL_FEATURES] = df[PAD20.NUMERICAL_FEATURES].replace("EMPTY", 0).astype(float)
    df[PAD20.TARGET_COLUMN] = df[PAD20.TARGET_COLUMN].cat.remove_categories('EMPTY')

    df.loc[:, ['background_father', 'background_mother']].replace('BRASIL', 'BRAZIL', inplace=True)
    df = pd.get_dummies(df, columns=PAD20.RAW_CATEGORICAL_FEATURES, dtype=np.int8)
    return df.drop(columns=[c for c in df.columns if c.endswith('EMPTY')])

if __name__ == '__main__':
    print("- Loading the dataset")
    df = pd.read_csv(config.PAD_20_RAW_METADATA)

    print("- Splitting the dataset...")
    print('- Grouping patients ids in the same folder')
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    df['folder'] = None
    for i, (_, test_indexes) in enumerate(kfold.split(df, df[PAD20.TARGET_COLUMN], groups=df[PAD20.PATIENT_ID])):
        df.loc[test_indexes, 'folder'] = i + 1

    # Validate patient id separation across folders
    patient_ids = df.groupby('folder')[PAD20.PATIENT_ID].unique()
    for i, ids in enumerate(patient_ids):
        for j, other_ids in enumerate(patient_ids):
            if i !=j and set(ids).intersection(other_ids):
                raise ValueError(f"Patient IDs {ids} and {other_ids} are present in the same folder {i+1} and {j+1}.")

    print("- Converting the labels to numbers")
    df = label_categorical_to_number (df, PAD20.TARGET_COLUMN, col_target_number=PAD20.TARGET_NUMBER_COLUMN)

    config.DATA_PATH.parent.mkdir(exist_ok=True)
    config.DATA_PATH.mkdir(exist_ok=True)

    simulate_missing_data(df, save_folder=config.DATA_PATH,
                            encoder_function=one_hot_encode,
                            encoder_name= f'one-hot',
                            features=PAD20.RAW_CATEGORICAL_FEATURES + PAD20.NUMERICAL_FEATURES,
                            dataset='pad-ufes-20')
