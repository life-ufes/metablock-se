import os
import time
import config
import pandas as pd

from pathlib import Path
from sacred import Experiment
from functools import partial
from model2vec import StaticModel
from sacred.observers import FileStorageObserver
from sentence_transformers import SentenceTransformer
from benchmarks.train_test_folder import train_test_folder
from benchmarks.pad20.image_augmentation import ImgTrainTransform, ImgEvalTransform

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TARGET_COLUMN = "diagnostic"
TARGET_NUMBER_COLUMN = "diagnostic_number"
IMG_COLUMN = "img_id"

_METADATA_COLUMNS = ["smoke_False","smoke_True","drink_False","drink_True","background_father_POMERANIA",
                     "background_father_GERMANY","background_father_BRAZIL","background_father_NETHERLANDS",
                     "background_father_ITALY","background_father_POLAND","background_father_UNK",
                     "background_father_PORTUGAL","background_father_BRASIL","background_father_CZECH",
                     "background_father_AUSTRIA","background_father_SPAIN","background_father_ISRAEL",
                     "background_mother_POMERANIA","background_mother_ITALY","background_mother_GERMANY",
                     "background_mother_BRAZIL","background_mother_UNK","background_mother_POLAND",
                     "background_mother_NORWAY","background_mother_PORTUGAL","background_mother_NETHERLANDS",
                     "background_mother_FRANCE","background_mother_SPAIN","age","pesticide_False","pesticide_True",
                     "gender_FEMALE","gender_MALE","skin_cancer_history_True","skin_cancer_history_False",
                     "cancer_history_True","cancer_history_False","has_piped_water_True","has_piped_water_False",
                     "has_sewage_system_True","has_sewage_system_False","fitspatrick_3.0","fitspatrick_1.0",
                     "fitspatrick_2.0","fitspatrick_4.0","fitspatrick_5.0","fitspatrick_6.0","region_ARM",
                     "region_NECK","region_FACE","region_HAND","region_FOREARM","region_CHEST","region_NOSE",
                     "region_THIGH","region_SCALP","region_EAR","region_BACK","region_FOOT","region_ABDOMEN",
                     "region_LIP","diameter_1","diameter_2","itch_False","itch_True","itch_UNK","grew_False",
                     "grew_True","grew_UNK","hurt_False","hurt_True","hurt_UNK","changed_False","changed_True",
                     "changed_UNK","bleed_False","bleed_True","bleed_UNK","elevation_False","elevation_True","elevation_UNK"]

# Starting sacred experiment
ex = Experiment()

######################################################################################

@ex.config
def cnfg():

    # Defines the folder to be used as validation
    _folder = 2

    _missing_percentage = 0
    # Models configurations
    _use_meta_data = True
    _neurons_reducer_block = 0
    _comb_method = 'metablock' # metanet, concat, or metablock
    _comb_config = [64, 81] # number of metadata
    _batch_size = 64
    _epochs = 100
    _model_name = 'resnet-50'
    _save_folder = f"benchmarks/pad20/results/{_model_name}_{_comb_method}_folder_{str(_folder)}_{str(time.time()).replace('.', '')}"

    # Training variables
    _best_metric = "loss"
    _pretrained = True
    _lr_init = 0.0001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None
    _weights = "frequency"
    _optimizer = 'adam'
    _append_observer = True
    _preprocessing = 'onehot'
    _llm_type = "sentence-transformers/paraphrase-albert-small-v2"

def get_sentence_transformer(_type):
    print(f"Loading encoder: {_type}...")
    return SentenceTransformer(_type)

def get_metadata(df, metadata_columns, sentence_model, sentences):
    return sentence_model.encode([sentences.loc[img]['sentence'] for img in df[IMG_COLUMN].values], show_progress_bar=False)

def _get_metadata_fn(file_path, _llm_type):
    return partial(get_metadata, sentence_model=get_sentence_transformer(_llm_type), 
                                   sentences=pd.read_csv(file_path, index_col='img_id'))

@ex.automain
def main (_folder, _lr_init, _sched_factor, _sched_min_lr, _sched_patience, _batch_size, _epochs, 
          _early_stop, _weights, _model_name, _pretrained, _save_folder, _best_metric, _optimizer,
          _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data, _metric_early_stop,
          _append_observer, _preprocessing, _llm_type, _missing_percentage):

    if _append_observer:
        ex.observers.append(FileStorageObserver(_save_folder))

    _get_metadata_fn = None
    if _preprocessing == 'onehot' or _preprocessing is None:
        file_path = config.DATA_PATH / f'pad-ufes-20-one-hot-missing-{_missing_percentage}.csv'
    else:
        file_path = config.DATA_PATH / f'pad-ufes-20-sentence-missing-{_missing_percentage}.csv'
        
        # function to generate embeddings from SBERT models
        _get_metadata_fn = partial(get_metadata, sentence_model=get_sentence_transformer(_llm_type), 
                                   sentences=pd.read_csv(file_path, index_col='img_id'))

    pd.read_csv(file_path).to_csv(Path(_save_folder).parent.parent / f'missing_{_missing_percentage}' / f'metadata-{"one-hot" if _preprocessing != "sentence-embedding" else "sentence"}.csv', index=False)

    train_test_folder(file_path, _folder, _lr_init, _sched_factor, _sched_min_lr, 
                    _sched_patience, _batch_size, _epochs, _early_stop, _weights, _model_name,
                    _pretrained, _optimizer, _save_folder, _best_metric, _neurons_reducer_block,
                    _comb_method, _comb_config, _use_meta_data, _metric_early_stop, IMG_COLUMN,
                    TARGET_COLUMN, TARGET_NUMBER_COLUMN, _METADATA_COLUMNS, config.PAD_20_IMAGES_FOLDER,
                    initial_weights_path=None, get_metadata_fn=_get_metadata_fn,
                    img_train_transform=ImgTrainTransform(), img_eval_transform=ImgEvalTransform())