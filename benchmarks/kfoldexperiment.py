import time

from sacred.observers import FileStorageObserver
from models.models_hub import CONFIG_METABLOCK_BY_MODEL
from benchmarks.pad20.folder_experiment import ex as experiment

def get_comb_method_and_preprocessing(_comb_method):
    if _comb_method:
        if _comb_method.endswith('-se'):
            return _comb_method[:-3], 'sentence-embedding'
        return _comb_method, 'onehot'
    return None, None

if __name__=="__main__":
    start_time = str(time.time()).replace('.', '')

    optimizer = 'adam'
    early_stop_metric = 'loss'
    missing_percentages = [0, 5, 10, 20, 30, 50, 70]
    feature_fusion_methods = ['metablock-se', None, 'metablock', ]
    models = [ 'mobilenet', 'caformer_s18', 'resnet-50', 'efficientnet-b4',] 

    training_info_folder = f'opt_{optimizer}_early_stop_{early_stop_metric}'
    for i, _comb_method in enumerate(feature_fusion_methods):
        _comb_method, _preprocessing = get_comb_method_and_preprocessing(_comb_method)

        comb_method_folder = f'{_comb_method}{"-se" if _preprocessing == "sentence-embedding" else ""}' if _comb_method else 'no_metadata'

        for model_name in models:
            for missing_percentage in missing_percentages:
                if _comb_method == None and missing_percentage != 0:
                    continue # there is no metadata to simulate missingness

                folder_name = f'{training_info_folder}/{start_time}/{comb_method_folder}/{model_name}/missing_{missing_percentage}'

                for folder in range(1, 6):
                    save_folder = f"benchmarks/pad20/results/{folder_name}/folder_{str(folder)}"

                    experiment.observers = []
                    experiment.observers.append(FileStorageObserver.create(save_folder))
                    
                    config = {
                        "_missing_percentage": missing_percentage,
                        "_use_meta_data": _comb_method is not None,
                        "_comb_method": _comb_method,
                        "_comb_config": [CONFIG_METABLOCK_BY_MODEL[model_name], 81 if _preprocessing == 'onehot' else 768] if _comb_method else None,
                        "_save_folder": save_folder,
                        "_folder": folder,
                        "_model_name": model_name,
                        "_sched_patience": 10,
                        "_early_stop": 15,
                        "_batch_size": 35, #TODO: change to 65
                        "_optimizer": optimizer,
                        "_epochs": 100,
                        '_lr_init': 0.0001,
                        '_sched_factor': 0.1,
                        '_sched_min_lr': 1e-6,
                        '_append_observer': False, # avoid duplicating sacred observer
                        '_preprocessing': _preprocessing, # sentence-embedding or one-hot
                        '_llm_type': "sentence-transformers/paraphrase-albert-small-v2",
                        '_best_metric': early_stop_metric,
                    }
                    experiment.run(config_updates=config)