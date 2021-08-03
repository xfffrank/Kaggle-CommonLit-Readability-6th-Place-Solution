import os
import yaml


class CFG:

    test_mode = False  # only train on the first fold
    device = [0]

    # model-based
    model_path = 'deepset/roberta-base-squad2'  # pretrained model path
    model_name = 'deepset/roberta-base-squad2'  # model name for saving models. It should correspond to the name in Huggingface.
    max_len = 256
    backbone_out = 'attention'  # pooler, last_hidden_state, cls_token, attention, mean_pooling
    attention_hidden_sz = None  # 512, None
    remove_dp_in_bert = True
    dropout = 0.3  # the dropout in the head
    reinit_layer = 3  # pooler, 3

    # training-based
    seed = 42  # for kfolds
    train_seed = 57  # for training-related parts, i.e. data shuffling, layer initialisation
    fix_data_order = False  # fix the order in each training epoch
    kfold_strategy = 'old'  # specified, old
    remove_suspicious_id = True  # remove similar texts with different target values
    specified_folds_path = '# provide the specific folds path when setting kfold_strategy = "specified" #'

    num_folds = 5
    batch_sz = 8
    accumulate_grad_batches = 1
    use_smart_batching = False
    epochs = 5
    warmup_ratio = 0.06
    learning_rate = 2e-5
    non_backbone_lr = learning_rate  # 1e-3, learning_rate
    use_grouped_params = False
    lr_epsilon = 0.95  # 0.95, 0.8
    decay = 'linear'
    wts_decay = 0.005 if 'large' in model_name else 1e-2  # 1e-2
    betas = (0.9, 0.98) if 'large' in model_name else (0.9, 0.999)   # default: (0.9, 0.999) 

    val_check_interval = 1.0  # 10, 5, 1.0
    delayed_val_check_ep = 2
    delayed_val_check_interval = 10
    swa = True
    precision = 32  # default: 32

    # environment
    output_dir = './outputs'  # provide a dir for saving models
    params_dir = './outputs'  # provide a dir for saving hyper-parameters used
    log_dir = './logs'  # provide a dir for logging
    env = 'local'  # kaggle, local
    progress_bar = 0 if env == 'kaggle' else 1
    verbose = False if env == 'kaggle' else True

    # program
    num_workers = 20

    @staticmethod
    def get_next_exp_id():
        os.makedirs(CFG.output_dir, exist_ok=True)
        os.makedirs(CFG.params_dir, exist_ok=True)
        os.makedirs(CFG.log_dir, exist_ok=True)
        if len(os.listdir(CFG.params_dir)) == 0: return 0
        last_exp_num = int(sorted(os.listdir(CFG.params_dir))[-1].split('_')[1])
        return last_exp_num + 1

    @staticmethod
    def save_config(exp_id):
        config_dict = {}
        for k, v in vars(CFG).items():
            if k.startswith('_') or isinstance(v, staticmethod): continue
            config_dict[k] = v
        save_dir = os.path.join(CFG.params_dir, f'exp_{str(exp_id).zfill(3)}')
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/hparams.yaml', 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)

