import os
import yaml


class CFG:

    # set up
    input_file = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/inputs/train.csv'  # Can be found in the competition website
    external_file = '/home/oatos/Documents/code/kaggle/Kaggle-CommonLit-6th-place-solution/external_data/external_all.csv'  # Can be found in the github repo
    output_dir = '/home/oatos/Documents/code/kaggle/Kaggle-CommonLit-6th-place-solution/outputs'  # provide a dir for saving models and hyperparameters
    log_dir = '/home/oatos/Documents/code/kaggle/Kaggle-CommonLit-6th-place-solution/logs'  # provide a directory for tensorboard logging
    pretrain_config_path = '/home/oatos/Documents/code/kaggle/Kaggle-CommonLit-6th-place-solution/training/pretrain_config.yaml'  # Can be found in the github repo
    finetune_config_path = '/home/oatos/Documents/code/kaggle/Kaggle-CommonLit-6th-place-solution/training/finetune_config.yaml'  # Can be found in the github repo

    # model-based
    model_path = 'test'  # pretrained model path
    model_name = 'test'  # model name for saving models. It should correspond to the name in Huggingface.
    max_len = 256
    backbone_out = 'attention'  # pooler, last_hidden_state, cls_token, attention, mean_pooling
    remove_dp_in_bert = False
    dropout = 0.3  # the dropout in the head
    reinit_layer = 0  # pooler, 3

    # training-based
    seed = 42  # for kfolds
    train_seed = 57  # for training-related parts, i.e. data shuffling, layer initialisation
    fix_data_order = False  # fix the order in each training epoch

    num_folds = 5
    batch_sz = 8
    accumulate_grad_batches = 1
    epochs = 5
    warmup_ratio = 0.06
    learning_rate = 2e-5
    decay = 'linear'
    wts_decay = 0.005 if 'large' in model_name else 1e-2  # 1e-2
    betas = (0.9, 0.98) if 'large' in model_name else (0.9, 0.999)   # default: (0.9, 0.999) 

    delayed_val_check_ep = 2  # start evaluation after n epochs
    delayed_val_check_interval = 10  # evaluation interval
    swa = True
    precision = 32  # default: 32

    # environment
    device = [0]
    env = 'local'  # kaggle, local
    progress_bar = 0 if env == 'kaggle' else 1
    verbose = False if env == 'kaggle' else True

    # program
    num_workers = 4

    @staticmethod
    def get_next_exp_id():
        if len(os.listdir(CFG.output_dir)) == 0: return 0
        last_exp_num = int(sorted(os.listdir(CFG.output_dir))[-1].split('_')[1])
        return last_exp_num + 1

    @staticmethod
    def save_config(exp_id):
        config_dict = {}
        for k, v in vars(CFG).items():
            if k.startswith('_') or isinstance(v, staticmethod): continue
            config_dict[k] = v
        save_dir = os.path.join(CFG.output_dir, f'exp_{str(exp_id).zfill(3)}')
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/hparams.yaml', 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)

