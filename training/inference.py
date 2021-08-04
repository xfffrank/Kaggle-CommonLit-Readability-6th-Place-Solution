'''
This is for testing the CV of a single model.
'''

from numpy.core.defchararray import mod
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, logging
logging.set_verbosity_error()
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import torch

import pandas as pd
import gc
import numpy as np
import os
from tqdm import tqdm
from utils import *
from networks import *
from config import CFG


def predict_dataloader(loader, model, return_features=False):
    model = model.cuda()
    model = model.eval()
    preds, feats = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(batch) == 3:
                ids, mask, y = batch
            else:
                ids, mask = batch
            ids = ids.to('cuda')
            mask = mask.to('cuda')
            if return_features:
                y_pred, feat_pred = model([ids, mask], return_features)
                y_pred = y_pred.detach().cpu().numpy()
                feat_pred = feat_pred.detach().cpu().numpy()
                feats.append(feat_pred)
            else:
                y_pred = model([ids, mask]).detach().cpu().numpy()
            preds.append(y_pred)
    preds = np.concatenate(preds)
    if return_features:
        feats = np.concatenate(feats)
        return preds, feats
    else:
        return preds


def compute_cv_and_predict_testset(train_df, test_df, sub, ckpt_root, compute_cv=True):
    oof = np.zeros_like(train_df.target, dtype=float)
    test_preds = np.zeros_like(sub.target, dtype=float)
    tokenizer = get_tokenizer(CFG.model_path)
    test_dataset = CommonLitDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_sz*8, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
    for k in range(CFG.num_folds):
        ckpt_path = f'{ckpt_root}/{CFG.model_name}_seed{CFG.seed}_fold{k}.ckpt'
        print(ckpt_path)
        try:
            model = CommonLitModel.load_from_checkpoint(ckpt_path)
        except:
            model = CommonLitModel.load_from_checkpoint(ckpt_path, train_steps=0, val_size=0)
        print('Checkpoint loaded!')

        # test
        preds = predict_dataloader(test_loader, model)
        test_preds += preds

        if compute_cv:
            val_data = train_df[train_df['kfold'] == k]
            val_dataset = CommonLitDataset(val_data, tokenizer)
            print(f'Fold {k + 1} val size: {len(val_dataset)}')
            val_loader = DataLoader(val_dataset, batch_size=CFG.batch_sz*8, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
            # validation
            preds = predict_dataloader(val_loader, model)
            oof[train_df['kfold'] == k] = preds
            val_score = RMSE(val_data.target.values, preds)
            msg = f'Best validation loss for fold {k+1}: {val_score}'
            print(msg)
        # clean cache
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if compute_cv:
        print(f'CV RMSE: {RMSE(oof, train_df.target.values)}')
    return oof, test_preds / CFG.num_folds


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        
def main(ckpt_root, save_oof=False, save_path=None):
    train_df = pd.read_csv('../inputs/train.csv')
    train_df = create_folds(train_df, num_splits=CFG.num_folds, random_seed=CFG.seed)
    
    test_df = pd.read_csv('../inputs/test.csv')
    sub = pd.read_csv('../inputs/sample_submission.csv')
    oof, test_preds = compute_cv_and_predict_testset(train_df, test_df, sub, ckpt_root)
    if save_oof:
        np.save(save_path, oof)
        print(f'[INFO] OOF result saved!')
    sub.target = test_preds


if __name__ == '__main__':
    CFG.model_name = 'ahotrod/electra_large_discriminator_squad2_512'
    CFG.model_path = 'ahotrod/electra_large_discriminator_squad2_512'
    CFG.backbone_out = 'attention'  # attention, mean_pooling
    CFG.remove_suspicious_id = True
    ckpt_path = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/outputs/exp_827_*'
    save_oof = False if CFG.remove_suspicious_id else True
    save_path = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/notebooks/oof_preds/exp_780_rl_sq2_mlm'
    main(ckpt_path, save_oof=save_oof, save_path=save_path)


    