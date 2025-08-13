import time
import pandas as pd

from pathlib import Path
from sacred import Experiment
from functools import partial
from benchmarks.pyrotrain import train
from config import PAD_20_BAYESIAN_DATA
from torch.utils.data import DataLoader
from benchmarks.pad20.bn.dataset import PAD20
from sacred.observers import FileStorageObserver
from benchmarks.pad20.bn.models import HeMaskedBayesianNetwork

def get_dataloader(model, filename, batch_size, features, split):
    if split not in ['train', 'val', 'test']:
        raise ValueError('The split should be one of: train, val, test.')

    return DataLoader(
        PAD20(PAD_20_BAYESIAN_DATA / model / filename, split=split, features=features),
        batch_size=batch_size,
        shuffle=split == 'train',
        pin_memory=True,
        num_workers=16
    )

ex = Experiment('pad20_bn')

@ex.config
def cnfg():
    # we used the same features as He et al. 
    features = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'region', 
                'diameter', 'age_group', 'ACK', 'BCC', 'NEV', 'MEL', 'SEK', 'SCC']
    
    folder = 1
    missing_percentage = 0
    epochs = 100
    batch_size = 64
    learning_rate = 2.5e-3
    early_stop_patience = 15
    early_stop_metric = 'bacc' # Following He et al, we used the balanced accuracy to select the best model
    model_name = 'resnet-50'
    append_observer = True
    save_folder = Path(f"benchmarks/pad20/results/bayesiannetwork/{str(time.time()).replace('.', '')}/{model_name}/folder_{str(folder)}")

@ex.automain
def main(features, epochs, batch_size, learning_rate, early_stop_patience, early_stop_metric, model_name, folder, missing_percentage, append_observer, save_folder):

    if append_observer:
        ex.observers.append(FileStorageObserver.create(save_folder))

    filename = f'folder_{folder}_missing_{missing_percentage}.csv' if missing_percentage is not None else f'folder_{folder}.csv'
    df = pd.read_csv(PAD_20_BAYESIAN_DATA / model_name / filename, index_col=0)

    # save metadata
    df.to_csv(save_folder / 'metadata.csv', index=False)

    train(HeMaskedBayesianNetwork(), partial(get_dataloader, model=model_name, filename=filename), 
          features=features, folder=folder, save_folder=save_folder, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
          early_stop_patience=early_stop_patience, early_stop_metric=early_stop_metric)