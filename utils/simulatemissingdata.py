
import os
import numpy as np
import pandas as pd

def remove_samples(df, missing_percentage, features):
    np.random.seed(42)
    mask = np.random.rand(len(df), len(features)) < (1 - missing_percentage)
    new_values = pd.Series([np.nan for _ in features], index=features)
    df[features] = df[features].where(mask, other=new_values, axis=1)
    return df

def simulate_missing_data(df, save_folder, features, dataset, encoder_function,
                            encoder_name:str, missing_percentages=[0, 5, 10, 20, 30, 50, 70],
                            save_function=None):
    os.makedirs(save_folder, exist_ok=True)
    all_features = []
    for missing_percentage in missing_percentages:

        df_with_missing = df.copy()
        if (missing_percentage > 0):
            df_with_missing = remove_samples(df_with_missing, missing_percentage/100, features)

        encoded = encoder_function(df_with_missing)
        if missing_percentage != 0:
            # makes sure every csv has the same columns
            for feature in all_features:
                if feature not in encoded.columns:
                    encoded[feature] = 0
        else:
            all_features = encoded.columns

        if save_function is not None:
            save_function(encoded, missing_percentage, False)
        else:
            encoded.to_csv(os.path.join(save_folder, f"{dataset}-{encoder_name}-missing-{missing_percentage}.csv"), index=False)
