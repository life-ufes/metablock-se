import os
import pyro
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyro.infer import SVI, TraceEnum_ELBO
from torchmetrics.classification import Accuracy, Recall, AUROC
from pprint import pprint
import logging

logging.basicConfig(level=logging.WARNING)

bacc = Accuracy(task='multiclass', num_classes=6, average='macro').cuda()
acc = Accuracy(task='multiclass', num_classes=6, average='micro').cuda()
auc = AUROC(task='multiclass', num_classes=6, average='macro').cuda()

def _evaluate(svi, model, dataloader, train=False):
    loss = 0
    preds_all = torch.empty(0).cuda()
    probs_all = torch.empty(0).cuda()
    labels_all = torch.empty(0).cuda()
    imgs = []
    for img, embeddings, labels in dataloader:
        embeddings, labels = embeddings.cuda(), labels.cuda()
        feature_args = embeddings.unbind(dim=1)
        if train:
            loss += svi.step(*feature_args, diagnosis_obs=labels)
        else:
            loss += svi.evaluate_loss(*feature_args, diagnosis_obs=labels)
        preds, probs = model.predict(*feature_args)
        preds_all = torch.cat((preds_all, preds))
        probs_all = torch.cat((probs_all, probs))
        labels_all = torch.cat((labels_all, labels))
        imgs.extend(img)
    normalizer = len(dataloader)
    epoch_loss = loss/normalizer
    return {
            'bacc': bacc(preds_all.long(), labels_all.long()),
            'auc': auc(probs_all, labels_all.long()),
            'acc': acc(preds_all.long(), labels_all.long()),
            'loss': epoch_loss,
        }, {
            'probs': probs_all,
            'preds': preds_all,
            'labels': labels_all,
            'imgs': imgs
        }

def _evaluate_and_update_params(svi, model, dataloader):
    return _evaluate(svi, model, dataloader, True)

def _has_improved(new_value, old_value, metric_name):
    return new_value < old_value if metric_name == 'loss' else new_value > old_value

def train(bayesian_network, get_dataloader, features, folder, save_folder, batch_size=64, epochs=100, learning_rate=2.5e-3, early_stop_patience=15, early_stop_metric='bacc'):
    metrics = { 'bacc': [], 'loss': [] }

    os.makedirs(save_folder / 'best_checkpoint', exist_ok=True)
    early_stop_counter = 0
    pyro.clear_param_store() 

    train = get_dataloader(batch_size=batch_size, features=features, split='train')
    val = get_dataloader(batch_size=batch_size, features=features, split='val')

    optimizer = pyro.optim.Adam({'lr':learning_rate})
    svi = SVI(model=bayesian_network.model, guide=bayesian_network.guide, optim=optimizer, loss=TraceEnum_ELBO())

    best_metrics = {}
    best_metric_value = 0 if early_stop_metric != 'loss' else np.inf
        
    progress_bar = tqdm(range(1, epochs+1),
                desc=f'Folder {folder}',
                dynamic_ncols=True,
                leave=False,
                unit='epoch')

    for epoch in progress_bar:
        train_metrics, _ = _evaluate_and_update_params(svi, bayesian_network, train)
        
        val_metrics, _ = _evaluate(svi, bayesian_network, val)

        train_postfix = {f'train_{metric}':f'{value:.3f}' for metric, value in train_metrics.items()}
        val_postfix = {metric:f'{value:.3f}' for metric, value in val_metrics.items()}

        progress_bar.set_postfix(train_postfix | val_postfix)

        if _has_improved(val_metrics[early_stop_metric], best_metric_value, early_stop_metric):
            best_metric_value = val_metrics[early_stop_metric]
            best_metrics = val_metrics
            early_stop_counter = 0
            pyro.get_param_store().save(save_folder / 'best_checkpoint' / 'model.pt')
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping on epoch {epoch}')
            break

    print('\nValidation best metrics:')
    pprint(best_metrics)