import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning when using smart batching
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
import yaml
from argparse import ArgumentParser

from config import CFG


def prepare_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        help="Configuration scheme",
        default=None,
    )

    args = parser.parse_args()
    print(f'[INFO] Using configuration for {args.config}')

    with open(CFG.finetune_config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg[args.config].items():
            setattr(CFG, k, v)


# reference: https://www.kaggle.com/abhishek/step-1-create-folds
def create_folds(df, num_splits, random_seed):
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


if CFG.env == 'local':
    '''
    Custom checkpoint class wrappers.
    '''
    from typing import Any, Optional, Union
    from pathlib import Path
    from pytorch_lightning.utilities import rank_zero_deprecation  # import error in kaggle
    from pytorch_lightning.utilities.types import STEP_OUTPUT

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



