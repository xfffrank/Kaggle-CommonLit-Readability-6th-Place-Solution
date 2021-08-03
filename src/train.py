from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, logging
logging.set_verbosity_warning()
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

import pandas as pd
import os
import gc
from config import CFG
from networks import *
from utils import *
from inference import predict_dataloader, RMSE


def train_kfold(df):
    if CFG.test_mode: print('='*20 + f' [INFO] Test mode on ' + '='*20)
    tokenizer = get_tokenizer(CFG.model_name)
    print(f'Model path: {CFG.model_path}')
    if CFG.env in ['local', 'colab']:
        exp_id = CFG.get_next_exp_id()
        CFG.save_config(exp_id)
        print('='*20 + f'Experiment ID: {exp_id}' + '='*20)
    
    oof = np.zeros_like(df.target, dtype=float)
    ckpt_dir = '/kaggle/working/' if CFG.env == 'kaggle' else os.path.join(CFG.output_dir, f'exp_{str(exp_id).zfill(3)}')

    for k in range(0,5):
        train_one_fold(df, oof, k, tokenizer, ckpt_dir, exp_id)
        if CFG.test_mode:
            break
    
    if not CFG.test_mode:
        msg = f'CV RMSE: {RMSE(df.target.values, oof)}'
        print(msg)
        if CFG.env in ['local', 'colab']:
            log_message(msg, exp_id)


def train_one_fold(df, oof, fold_num, tokenizer, ckpt_dir, exp_id):
    k = fold_num
    print('='*20 + f'Fold {k+1} training starts' + '='*20)
    seed_everything(CFG.train_seed + k)
    train_data = df[df['kfold'] != k]
    val_data = df[df['kfold'] == k]

    if CFG.use_smart_batching:
        print('[INFO] Using smart batching...')
        train_dataset = SmartBatchingDataset(train_data, tokenizer)
        sampler = SmartBatchingSampler(
            data_source=train_dataset._data,
            batch_size=CFG.batch_sz
        )
        collate_fn = SmartBatchingCollate(
            targets=train_dataset._targets,
            max_length=CFG.max_len,
            pad_token_id=tokenizer.pad_token_id
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=CFG.batch_sz,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=CFG.num_workers,
            pin_memory=True
        )
    else:
        train_dataset = CommonLitDataset(train_data, tokenizer, shuffle=CFG.fix_data_order)
        # use shuffle=False to fix data order in each epoch
        epoch_shuffle = not CFG.fix_data_order
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_sz, shuffle=epoch_shuffle, num_workers=CFG.num_workers, pin_memory=True)     
    val_dataset = CommonLitDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_sz*4, shuffle=False, num_workers=CFG.num_workers, pin_memory=False)
    print(f'Train size: {len(train_dataset)}')
    print(f'Val size: {len(val_dataset)}')
    train_steps = len(train_loader)
    val_size = len(val_dataset)
    
    ckpt_filename = f'{CFG.model_name}_seed{CFG.seed}_fold{k}'
    callbacks = [
        CustomModelCheckpointDelayedEval(
            dirpath=ckpt_dir,
            filename=ckpt_filename,
            monitor='val_loss',
            verbose=CFG.verbose,
            mode='min',
            save_weights_only=True,
            train_steps=train_steps
        ),
    ]

    log_fold = 0
    if CFG.env == 'local' and CFG.decay != 'constant' and k == log_fold:
        print('[INFO] Logging learning rates only for the first fold.')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    if k == log_fold:
        logger = None if CFG.env in ['kaggle', 'colab'] else TensorBoardLogger(save_dir=CFG.log_dir, name='logs', version=f'exp_{str(exp_id).zfill(3)}')
    else:
        logger = None

    model = CommonLitModel(train_steps, val_size)
    print('Model created!')
    trainer = pl.Trainer(gpus=CFG.device, callbacks=callbacks, max_epochs=CFG.epochs, num_sanity_val_steps=2, logger=logger, val_check_interval=CFG.val_check_interval, progress_bar_refresh_rate=CFG.progress_bar, accumulate_grad_batches=CFG.accumulate_grad_batches, precision=CFG.precision, stochastic_weight_avg=CFG.swa)
    print('Start training...')
    trainer.fit(model, train_loader, val_loader)

    # clean cache
    del trainer, model
    gc.collect()

    # predict val data
    print('Loading the best checkpoint...')
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename) + '.ckpt' 
    model = CommonLitModel.load_from_checkpoint(ckpt_path)
    preds = predict_dataloader(val_loader, model)
    oof[df['kfold'] == k] = preds
    val_score = RMSE(df[df['kfold'] == k].target.values, preds)
    msg = f'Best validation loss for fold {k+1}: {val_score}'
    print(msg)

    # clean cache
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if CFG.env in ['local', 'colab']:
        log_message(msg, exp_id)


def main():
    train_df = pd.read_csv('../inputs/train.csv')
    train_df = create_folds(train_df, num_splits=CFG.num_folds, random_seed=CFG.seed)
    train_kfold(train_df)


if __name__ == '__main__':
    main()


    