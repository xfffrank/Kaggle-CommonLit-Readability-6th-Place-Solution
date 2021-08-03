import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning when using smart batching
import pandas as pd
import random
import more_itertools

import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import transformers
from transformers import AutoTokenizer
from sklearn import preprocessing
import re

from config import CFG


def create_folds(df, num_splits, random_seed):
    print(f'[INFO] Kfold strategy: {CFG.kfold_strategy}')
    if CFG.kfold_strategy == 'old':
        df = create_folds_old(df, num_splits, random_seed)
        if CFG.remove_suspicious_id:
            print('[INFO] Removing suspicious IDs...')
            df = remove_suspicious_id(df)
        return df
    elif CFG.kfold_strategy == 'specified':
        return pd.read_csv(CFG.specified_folds_path)
    else:
        raise NotImplementedError()


def create_folds_old(df, num_splits, random_seed):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(df))))
    
    # Bin values into discrete intervals.
    df.loc[:, "bins"] = pd.cut(
        df["target"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, 'kfold'] = f
    
    # drop the bins column
    # df = df.drop("bins", axis=1)

    # return dfframe with folds
    return df


def get_tokenizer(model_name_or_path):
    try:
        print('[INFO] Using cached tokenizer...')
        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    except:
        print('[INFO] Downloading tokenizer...')
        return AutoTokenizer.from_pretrained(model_name_or_path)


def remove_suspicious_id(df):
    '''
    Remove similar texts with different target values.
    '''
    ids_to_remove = ['da2dbbc70', '0684bb254', '0e6bada1d']
    df = df[df['id'].apply(lambda x: x not in ids_to_remove)]
    return df


class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, shuffle=False):
        self.df = df
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=CFG.train_seed).reset_index(drop=True)
        self.labeled = 'target' in df.columns
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        text = item['excerpt']
        token = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=CFG.max_len)
        if self.labeled:
            target = item['target']
            target = torch.tensor(target, dtype=torch.float)
            return token['input_ids'].squeeze(), token['attention_mask'].squeeze(), target
        else:
            return token['input_ids'].squeeze(), token['attention_mask'].squeeze()


def log_message(msg, exp_id):
    dir_path = [CFG.output_dir, CFG.params_dir] if CFG.env == 'colab' else [CFG.output_dir]
    for path in dir_path:
        log_file = os.path.join(path, f'exp_{str(exp_id).zfill(3)}', f'{CFG.model_name}.txt')
        with open(log_file, 'a') as f:
            f.write(msg + '\n')


# credit: https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(SmartBatchingDataset, self).__init__()
        self.tokenizer = tokenizer
        self._data = (
            f"{self.tokenizer.bos_token} " + df.excerpt + f" {self.tokenizer.eos_token}" 
        ).apply(self.tokenizer.tokenize).apply(self.tokenizer.convert_tokens_to_ids).to_list()
        self._targets = None
        if 'target' in df.columns:
            self._targets = df.target.tolist()
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]


# credit: https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
    
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds


# credit: https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies/notebook
class SmartBatchingCollate:
    '''
    SmartBatchingCollate will add padding upto highest sequence length, make attention masks, targets for each sample in batch.
    '''
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id
        
    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)
        
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        
        if self._targets is not None:
            output = input_ids, attention_mask, torch.tensor(targets)
        else:
            output = input_ids, attention_mask
        return output
    
    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks


if CFG.env == 'local':
    '''
    Custom checkpoint class wrappers.
    '''
    from typing import Any, Callable, Dict, Optional, Union
    from pathlib import Path
    from pytorch_lightning.utilities import rank_zero_deprecation  # import error in kaggle
    from pytorch_lightning.utilities.types import _METRIC, STEP_OUTPUT
    class CustomModelCheckpoint(ModelCheckpoint):

        def __init__(
            self,
            dirpath: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            monitor: Optional[str] = None,
            verbose: bool = False,
            save_last: Optional[bool] = None,
            save_top_k: Optional[int] = None,
            save_weights_only: bool = False,
            mode: str = "min",
            auto_insert_metric_name: bool = True,
            every_n_train_steps: Optional[int] = None,
            every_n_val_epochs: Optional[int] = None,
            period: Optional[int] = None,
        ):
            super().__init__(dirpath=dirpath, filename=filename, monitor=monitor, verbose=verbose, save_last=save_last, save_top_k=save_top_k, save_weights_only=save_weights_only, mode=mode, auto_insert_metric_name=auto_insert_metric_name, every_n_train_steps=every_n_train_steps, every_n_val_epochs=every_n_val_epochs, period=period)
            self.eval_schedule = CFG.eval_schedule
            self.last_eval_step = 0
            self.eval_interval = self.eval_schedule[0][1]
            print(f'[INFO] Eval interval set to {self.eval_interval}')

        def on_train_batch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
        ) -> None:
            """ Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps` """
            if self._should_skip_saving_checkpoint(trainer):
                return
            step = trainer.global_step
            if step >= self.last_eval_step + self.eval_interval:
                # num = step - self.last_eval_step
                # print(f'Globel step: {step}')
                # print(f'Last step: {self.last_eval_step}')
                # print(f'[INFO] Num of steps elapsed: {num}')
                self.last_eval_step = step
                trainer.run_evaluation()
                # what can be monitored
                epoch = trainer.current_epoch
                global_step = trainer.global_step
                monitor_candidates = self._monitor_candidates(trainer, epoch=epoch, step=global_step)
                # custom code logic (Feng Xie)
                val_rmse = monitor_candidates['val_loss'].detach().cpu().numpy()
                for rmse, interval in self.eval_schedule:
                    if val_rmse >= rmse:
                        tmp_interval = interval
                        break
                if tmp_interval != self.eval_interval:
                    self.eval_interval = tmp_interval
                    # print(f'Current val rmse: {val_rmse}')
                    print(f'[INFO] Eval interval changed to {self.eval_interval}')


    class CustomModelCheckpointDelayedEval(ModelCheckpoint):

        def __init__(
            self,
            dirpath: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            monitor: Optional[str] = None,
            verbose: bool = False,
            save_last: Optional[bool] = None,
            save_top_k: Optional[int] = None,
            save_weights_only: bool = False,
            mode: str = "min",
            auto_insert_metric_name: bool = True,
            every_n_train_steps: Optional[int] = None,
            every_n_val_epochs: Optional[int] = None,
            period: Optional[int] = None,
            train_steps = 0
        ):
            super().__init__(dirpath=dirpath, filename=filename, monitor=monitor, verbose=verbose, save_last=save_last, save_top_k=save_top_k, save_weights_only=save_weights_only, mode=mode, auto_insert_metric_name=auto_insert_metric_name, every_n_train_steps=every_n_train_steps, every_n_val_epochs=every_n_val_epochs, period=period)
            # self.eval_schedule = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
            self.eval_interval = CFG.delayed_val_check_interval
            self.last_eval_step = 0
            # make sure the result is consistant with different `delayed_val_check_ep`
            self.delayed_steps = (int(CFG.delayed_val_check_ep * train_steps) // self.eval_interval) * self.eval_interval
            print(f'[INFO] Delayed steps before evaluation: {self.delayed_steps}')
            self.val_check_mode = False

        def on_train_batch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
        ) -> None:
            """ Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps` """
            if self._should_skip_saving_checkpoint(trainer):
                return
            step = trainer.global_step
            if step == self.delayed_steps:
                self.val_check_mode = True
                self.last_eval_step = step
                print('[INFO] The val check mode is turned on!')

            if self.val_check_mode and step == self.last_eval_step + self.eval_interval:
                self.last_eval_step = step
                trainer.run_evaluation()



