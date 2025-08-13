from benchmarks.pad20.bn.train import ex as experiment_pad_20
from sacred.observers import FileStorageObserver
import time
import argparse
from pathlib import Path
if __name__=="__main__":
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run kfold validation')
    parser.add_argument('--missing', action='store_true', default=True)
    
    args = parser.parse_args()
    benchmark = 'pad20'

    experiment = experiment_pad_20

    start_time = str(time.time()).replace('.', '')
    models = ['resnet-50', 'mobilenet', 'efficientnet-b4', 'caformer_s18'] 
    missing_percentages = [0, 5, 10, 20, 30, 50, 70] if args.missing else [None]
    for model_name in models:
        for missing_percentage in missing_percentages:
            folder_name = f'bayesiannetwork/{start_time}/{model_name}'
            if missing_percentage is not None:
                folder_name = f'{folder_name}/missing_{missing_percentage}'

            for folder in range(1, 6):
                save_folder = f"benchmarks/{benchmark}/results/{folder_name}/folder_{str(folder)}"

                experiment.observers = []
                experiment.observers.append(FileStorageObserver.create(save_folder))
                config = {
                    "missing_percentage": missing_percentage,
                    "save_folder": Path(save_folder),
                    "folder": folder,
                    "model_name": model_name,
                    "early_stop_patience": 15,
                    "batch_size": 64,
                    "epochs": 100,
                    'learning_rate': 2.5e-3,
                    'append_observer': False,
                    'early_stop_metric': 'bacc', # used the same early_stop_metric as He et al.
                }
                experiment.run(config_updates=config)