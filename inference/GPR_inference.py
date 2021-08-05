import re
import os
import gc
import sys
import math
import time
import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from joblib import dump, load
import gc

import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (AutoModel, AutoConfig, AutoTokenizer)

from GPR_base import get_pyro_emb_preds

train_data = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
test_data = pd.read_csv('../input/commonlitreadabilityprize/test.csv')
sample = pd.read_csv('../input/commonlitreadabilityprize/sample_submission.csv')

target = train_data['target'].to_numpy()

def rmse_score(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

device = torch.device("cuda")
COMPUTE_CV = False

if COMPUTE_CV:
    test = train_data
    is_test = False
    mode = "train"
else:
    test = test_data
    is_test = True
    mode = "test"

config = {
    'batch_size':512,
    'max_len':256,
    'seed':42,
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=config['seed'])

class CLRPDataset(nn.Module):
    def __init__(self,df,tokenizer,max_len=250):
        self.excerpt = df['excerpt'].to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __getitem__(self,idx):
        encode = self.tokenizer(self.excerpt[idx],
                                return_tensors='pt',
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True)  
        return encode
    
    def __len__(self):
        return len(self.excerpt)
    
import os
import yaml
import pytorch_lightning as pl

class CFG:

    test_mode = False

    # model-based
    model_path = '../input/robertalarge-epoch5-textaugment'  # pretrained model path, {bert-base-uncased, roberta-base, roberta-large}
    model_name = 'roberta-large'  # model name for saving models. It should correspond to the name in Huggingface.
    max_len = 256
    backbone_out = 'attention'  # pooler, last_hidden_state, cls_token, attention, conv1d
    dropout = 0.5
    aug = False
    reinit_layer = 3

    # training-based
    seed = 42  # for kfolds
    train_seed = 42  # for training-related parts, i.e. data shuffling, layer initialisation
    fix_data_order = True
    kfold_strategy = 'random'  # random, old, new
    random_folds_path = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/inputs/external/random_folds.csv'

    num_folds = 5
    batch_sz = 8
    accumulate_grad_batches = 1
    use_smart_batching = False
    epochs = 5
    es_epoch = 2  # early stopping
    warmup_ratio = 0
    learning_rate = 2e-5
    last_linear_lr = learning_rate  # 1e-3, learning_rate
    use_grouped_params = False
    lr_epsilon = 0.95
    decay = 'linear'
    wts_decay = 0.005 if 'large' in model_name else 1e-2  # 1e-2
    betas = (0.9, 0.98) if 'large' in model_name else (0.9, 0.999)   # default: (0.9, 0.999) 

    val_check_interval = 10  # 10, 5, 1.0
    swa = False

    # inference
    num_tta = 2

    # environment
    output_dir = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/outputs'
    params_dir = '/home/oatos/Documents/code/kaggle/CommonLit-Readability/outputs'
    log_dir = '/home/oatos/Documents/code/kaggle/CommonLit-Readability'
    device = [0]
    env = 'local'  # kaggle, local
    progress_bar = 0 if env == 'kaggle' else 1
    verbose = False if env == 'kaggle' else True

    # program
    use_tpu = False
    num_workers = 20

    @staticmethod
    def get_next_exp_id():
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
            

def predict_dataloader(loader, model, output_features=False):
    preds = []
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(batch) == 3:
                ids, mask, y = batch
            else:
                ids, mask = batch
            ids = ids.to('cuda')
            mask = mask.to('cuda')
            if output_features:
                y_pred = model.extract_feats([ids, mask]).detach().cpu().numpy()
            else:
                y_pred, embedding = model([ids, mask])
                y_pred = y_pred.detach().cpu().numpy()
                embedding = embedding.detach().cpu().numpy()
            preds.append(y_pred)
            embeddings.append(embedding)
    preds = np.concatenate(preds)
    embeddings = np.concatenate(embeddings)
    return preds, embeddings

class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, shuffle=False):
        self.df = df
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)
        self.labeled = 'target' in df.columns
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        text = item['excerpt']
        token = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=CFG.max_len)
        if self.labeled:
            target = torch.tensor(item['target'], dtype=torch.float)
            return token['input_ids'].squeeze(), token['attention_mask'].squeeze(), target
        else:
            return token['input_ids'].squeeze(), token['attention_mask'].squeeze()

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

    
class CommonLitModel(pl.LightningModule):
    
    def __init__(self, train_steps):
        super().__init__()
        self.train_steps = train_steps
        self.config = self.get_config()
        self.backbone = self.get_backbone(self.config)
        self.num_labels = 1
        self.config.update({'num_labels': self.num_labels})
        self.dropout = nn.Dropout(CFG.dropout)
        if CFG.backbone_out == 'conv1d':
            self.clf = nn.Linear(1024, self.num_labels)
        else:
            self.clf = nn.Linear(self.config.hidden_size, self.num_labels)
        if CFG.backbone_out == 'attention':
            hidden_size = self.config.hidden_size
            self.head = AttentionHead(self.config.hidden_size, hidden_size) # self.config.hidden_size
        if CFG.backbone_out == 'conv1d':
            self.cnn1 = nn.Conv1d(self.config.hidden_size, 256, kernel_size=3, padding=2)
            self.cnn2 = nn.Conv1d(256, 512, kernel_size=3, padding=2)
            self.cnn3 = nn.Conv1d(512, 256, kernel_size=3, padding=2)
        if 'large' in CFG.model_name:
            self.layer_norm = nn.LayerNorm(self.config.hidden_size)
            self._init_weights(self.clf)
            self._init_weights(self.layer_norm)

    def get_config(self):
        return AutoConfig.from_pretrained(CFG.model_path)

    def _init_weights(self, module):
        initializer_range = 0.02
        init_res = False
        if isinstance(module, nn.Linear):
            init_res = True
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            init_res = True
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            init_res = True
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        return init_res

    def get_backbone(self, config):
        model = AutoModel.from_pretrained(CFG.model_path, config=config)
        # print(model)
        # raise
        if CFG.reinit_layer == 'pooler':
            print('Reinitializing Pooler Layer ...')
            encoder_temp = model
            encoder_temp.pooler.weight.data.normal_(mean=0.0, std=config.initializer_range)
            encoder_temp.pooler.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
            print('Done.!')
        elif CFG.reinit_layer > 0:
            # _model_type = 'roberta'
            print(f'Reinitializing Last {CFG.reinit_layer} Layers ...')
            # print(model)
            # encoder_temp = getattr(model, _model_type)
            num_modules = 0
            if 'bart' in CFG.model_name:
                temp = model.decoder
                for layer in temp.layers[-CFG.reinit_layer:]:
                    for module in layer.modules():
                        if self._init_weights(module): num_modules += 1
            elif 'xlnet' in CFG.model_name:
                temp = model
                for layer in temp.layer[-CFG.reinit_layer:]:
                    for module in layer.modules():
                        if self._init_weights(module): num_modules += 1
            else:
                temp = model.encoder
                for layer in temp.layer[-CFG.reinit_layer:]:
                    for module in layer.modules():
                        if self._init_weights(module): num_modules += 1
            print(f'Done reinitialising {num_modules} modules!')
        return model

    def extract_feats(self, inputs):
        input_ids, mask = inputs
        out = self.backbone(input_ids=input_ids, attention_mask=mask)
        
        if CFG.backbone_out == 'pooler':
            out = out['pooler_output']
        elif CFG.backbone_out == 'last_hidden_state':
            out = out['last_hidden_state']
            out = torch.mean(out, dim=1)
        elif CFG.backbone_out == 'cls_token':
            out = out['last_hidden_state'][:,0,:]
        elif CFG.backbone_out == 'attention':
            if 'pooler_output' in out.keys():
                pooler = out['pooler_output'].reshape((out['pooler_output'].shape[0], 1, self.config.hidden_size))
                x = torch.cat([out['last_hidden_state'], pooler],axis=1)  # (batch size, sequence len + 1, hidden size)
                # pooler_mask = torch.ones((out['pooler_output'].shape[0], 1)).to(f'cuda:{CFG.device[0]}')
                # mask = torch.cat([mask, pooler_mask], axis=1)
            else:
                x = out['last_hidden_state'] 
            # return last 12 layers
#             last_no_layers = -4
#             hidden_states = out['hidden_states']
#             pooled_output = torch.cat(tuple([hidden_states[i] for i in np.arange(last_no_layers, 0)]), dim=-1) # [-4, -3, -2, -1]
#             pooled_output = pooled_output[:, 0, :]

            out = self.head(x)
        elif CFG.backbone_out == 'conv1d':
            x = out['last_hidden_state']
            x = x.permute(0, 2, 1)  # (batch_sz, seq_len, hidden_size) -> (batch_sz, hidden_size, seq_len), channel first in pytorch
            x = self.cnn1(x)
            x = F.relu(x)  # (batch_sz, 256, seq_len)
            x = F.avg_pool1d(x, kernel_size=4)
            x = self.cnn2(x)
            x = F.relu(x)  # (batch_sz, 256, seq_len)
            x = F.avg_pool1d(x, kernel_size=4)
            x = self.cnn3(x)
            x = F.relu(x)  # (batch_sz, 256, seq_len)
            x = F.avg_pool1d(x, kernel_size=4)
            out = torch.flatten(x, start_dim=1)
        
        return out
    
    def forward(self, inputs):
        out = self.extract_feats(inputs)
        pooled_output = out
        if 'large' in CFG.model_name:
            out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.clf(out).squeeze()
        return out, pooled_output
    
def run_inference_fx(fold_df, model_name, ckpt_path, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fold_dataset = CommonLitDataset(fold_df, tokenizer)
    fold_loader = DataLoader(fold_dataset, batch_size=batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    ckpt_path = ckpt_path
    model_fx = CommonLitModel.load_from_checkpoint(ckpt_path, train_steps=0)
    
    model_fx.cuda()
    model_fx.eval()

    preds, embeddings = predict_dataloader(fold_loader, model_fx)

    del model_fx
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return preds, embeddings

if __name__ == '__main__':
    ## Load embeddings
    model_path = 'roberta-large'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'roberta-large'
    batch_size = 128
    preds_1_roberta_l_fx_old, embeddings1_rl_fx_old = run_inference_fx(test, model_path, '../input/fx_weights/exp_180/exp_180_*/roberta-large_seed42_fold0.ckpt', batch_size)
    preds_2_roberta_l_fx_old, embeddings2_rl_fx_old = run_inference_fx(test, model_path, '../input/fx_weights/exp_180/exp_180_*/roberta-large_seed42_fold1.ckpt', batch_size)
    preds_3_roberta_l_fx_old, embeddings3_rl_fx_old = run_inference_fx(test, model_path, '../input/fx_weights/exp_180/exp_180_*/roberta-large_seed42_fold2.ckpt', batch_size)
    preds_4_roberta_l_fx_old, embeddings4_rl_fx_old = run_inference_fx(test, model_path, '../input/fx_weights/exp_180/exp_180_*/roberta-large_seed42_fold3.ckpt', batch_size)
    preds_5_roberta_l_fx_old, embeddings5_rl_fx_old = run_inference_fx(test, model_path, '../input/fx_weights/exp_180/exp_180_*/roberta-large_seed42_fold4.ckpt', batch_size)
    preds_roberta_l_fx_old = (preds_1_roberta_l_fx_old + preds_2_roberta_l_fx_old + preds_3_roberta_l_fx_old + preds_4_roberta_l_fx_old + preds_5_roberta_l_fx_old) / 5


    model_path = 'microsoft/deberta-large'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'microsoft/deberta-large'
    batch_size = 128
    preds_1_deberta_l_fx, embeddings1_dl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold0.ckpt', batch_size)
    preds_2_deberta_l_fx, embeddings2_dl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold1.ckpt', batch_size)
    preds_3_deberta_l_fx, embeddings3_dl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold2.ckpt', batch_size)
    preds_4_deberta_l_fx, embeddings4_dl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold3.ckpt', batch_size)
    preds_5_deberta_l_fx, embeddings5_dl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold4.ckpt', batch_size)
    preds_deberta_l_fx = (preds_1_deberta_l_fx + preds_2_deberta_l_fx + preds_3_deberta_l_fx + preds_4_deberta_l_fx + preds_5_deberta_l_fx) / 5

    model_path = 'xlnet-large-cased'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'xlnet-large-cased'
    batch_size = 128
    preds_1_xlnet_l_fx, embeddings1_xl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_284_/exp_284/xlnet-large-cased_seed42_fold0.ckpt', batch_size)
    preds_2_xlnet_l_fx, embeddings2_xl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_284_/exp_284/xlnet-large-cased_seed42_fold1.ckpt', batch_size)
    preds_3_xlnet_l_fx, embeddings3_xl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_284_/exp_284/xlnet-large-cased_seed42_fold2.ckpt', batch_size)
    preds_4_xlnet_l_fx, embeddings4_xl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_284_/exp_284/xlnet-large-cased_seed42_fold3.ckpt', batch_size)
    preds_5_xlnet_l_fx, embeddings5_xl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_284_/exp_284/xlnet-large-cased_seed42_fold4.ckpt', batch_size)
    preds_xlnet_l_fx = (preds_1_xlnet_l_fx + preds_2_xlnet_l_fx + preds_3_xlnet_l_fx + preds_4_xlnet_l_fx + preds_5_xlnet_l_fx) / 5

    # model_path = '../input/robertalarge-epoch5-textaugment'
    model_path = 'roberta-large'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'roberta-large'
    batch_size = 128
    preds_1_roberta_l_s2_fx, embeddings1_rl_sq2_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold0.ckpt', batch_size)
    preds_2_roberta_l_s2_fx, embeddings2_rl_sq2_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold1.ckpt', batch_size)
    preds_3_roberta_l_s2_fx, embeddings3_rl_sq2_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold2.ckpt', batch_size)
    preds_4_roberta_l_s2_fx, embeddings4_rl_sq2_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold3.ckpt', batch_size)
    preds_5_roberta_l_s2_fx, embeddings5_rl_sq2_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold4.ckpt', batch_size)
    preds_roberta_l_s2_fx = (preds_1_roberta_l_s2_fx + preds_2_roberta_l_s2_fx + preds_3_roberta_l_s2_fx + preds_4_roberta_l_s2_fx + preds_5_roberta_l_s2_fx) / 5

    model_path = 'allenai/longformer-large-4096-finetuned-triviaqa'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'allenai/longformer-large-4096-finetuned-triviaqa'
    batch_size = 64
    preds_1_longformer_l_tqa_fx, embeddings1_longformer_tqa_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold0.ckpt', batch_size)
    preds_2_longformer_l_tqa_fx, embeddings2_longformer_tqa_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold1.ckpt', batch_size)
    preds_3_longformer_l_tqa_fx, embeddings3_longformer_tqa_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold2.ckpt', batch_size)
    preds_4_longformer_l_tqa_fx, embeddings4_longformer_tqa_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold3.ckpt', batch_size)
    preds_5_longformer_l_tqa_fx, embeddings5_longformer_tqa_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold4.ckpt', batch_size)
    preds_longformer_ltqa_fx = (preds_1_longformer_l_tqa_fx + preds_2_longformer_l_tqa_fx + preds_3_longformer_l_tqa_fx + preds_4_longformer_l_tqa_fx + preds_5_longformer_l_tqa_fx) / 5

    model_path = 'roberta-large'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'roberta-large'
    batch_size = 128
    preds_1_roberta_l_s2_mlm_only_pl_fx, embeddings1_rl_sq2_mlm_only_pl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold0.ckpt', batch_size)
    preds_2_roberta_l_s2_mlm_only_pl_fx, embeddings2_rl_sq2_mlm_only_pl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold1.ckpt', batch_size)
    preds_3_roberta_l_s2_mlm_only_pl_fx, embeddings3_rl_sq2_mlm_only_pl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold2.ckpt', batch_size)
    preds_4_roberta_l_s2_mlm_only_pl_fx, embeddings4_rl_sq2_mlm_only_pl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold3.ckpt', batch_size)
    preds_5_roberta_l_s2_mlm_only_pl_fx, embeddings5_rl_sq2_mlm_only_pl_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold4.ckpt', batch_size)
    preds_roberta_l_s2_mlm_only_pl_fx = (preds_1_roberta_l_s2_mlm_only_pl_fx + preds_2_roberta_l_s2_mlm_only_pl_fx + preds_3_roberta_l_s2_mlm_only_pl_fx + preds_4_roberta_l_s2_mlm_only_pl_fx + preds_5_roberta_l_s2_mlm_only_pl_fx) / 5


    model_path = 'valhalla/bart-large-finetuned-squadv1'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'valhalla/bart-large-finetuned-squadv1'
    batch_size = 128
    preds_1_bl_mlm_rmdp_fx, embeddings1_bl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold0.ckpt', batch_size)
    preds_2_bl_mlm_rmdp_fx, embeddings2_bl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold1.ckpt', batch_size)
    preds_3_bl_mlm_rmdp_fx, embeddings3_bl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold2.ckpt', batch_size)
    preds_4_bl_mlm_rmdp_fx, embeddings4_bl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold3.ckpt', batch_size)
    preds_5_bl_mlm_rmdp_fx, embeddings5_bl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold4.ckpt', batch_size)
    preds_bl_mlm_rmdp_fx = (preds_1_bl_mlm_rmdp_fx + preds_2_bl_mlm_rmdp_fx + preds_3_bl_mlm_rmdp_fx + preds_4_bl_mlm_rmdp_fx + preds_5_bl_mlm_rmdp_fx) / 5


    model_path = 'microsoft/deberta-large'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'microsoft/deberta-large'
    batch_size = 128
    preds_1_deberta_l_mlm_rmdp_fx, embeddings1_dl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold0.ckpt', batch_size)
    preds_2_deberta_l_mlm_rmdp_fx, embeddings2_dl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold1.ckpt', batch_size)
    preds_3_deberta_l_mlm_rmdp_fx, embeddings3_dl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold2.ckpt', batch_size)
    preds_4_deberta_l_mlm_rmdp_fx, embeddings4_dl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold3.ckpt', batch_size)
    preds_5_deberta_l_mlm_rmdp_fx, embeddings5_dl_mlm_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold4.ckpt', batch_size)
    preds_deberta_l_mlm_rmdp_fx = (preds_1_deberta_l_mlm_rmdp_fx + preds_2_deberta_l_mlm_rmdp_fx + preds_3_deberta_l_mlm_rmdp_fx + preds_4_deberta_l_mlm_rmdp_fx + preds_5_deberta_l_mlm_rmdp_fx) / 5

    model_path = 'ahotrod/electra_large_discriminator_squad2_512'
    print("Loading {} embeddings".format(model_path))

    CFG.model_path = model_path
    CFG.model_name = 'ahotrod/electra_large_discriminator_squad2_512'
    batch_size = 128
    preds_1_el_s2_rmdp_fx, embeddings1_el_s2_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold0.ckpt', batch_size)
    preds_2_el_s2_rmdp_fx, embeddings2_el_s2_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold1.ckpt', batch_size)
    preds_3_el_s2_rmdp_fx, embeddings3_el_s2_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold2.ckpt', batch_size)
    preds_4_el_s2_rmdp_fx, embeddings4_el_s2_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold3.ckpt', batch_size)
    preds_5_el_s2_rmdp_fx, embeddings5_el_s2_rmdp_fx = run_inference_fx(test, model_path, '../input/fx_weights/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold4.ckpt', batch_size)
    preds_el_s2_rmdp_fx = (preds_1_el_s2_rmdp_fx + preds_2_el_s2_rmdp_fx + preds_3_el_s2_rmdp_fx + preds_4_el_s2_rmdp_fx + preds_5_el_s2_rmdp_fx) / 5


    embeddings1 = np.concatenate((normalize(embeddings1_longformer_tqa_fx), normalize(embeddings1_bl_mlm_rmdp_fx), normalize(embeddings1_rl_sq2_mlm_only_pl_fx),
                                  normalize(embeddings1_dl_mlm_rmdp_fx), normalize(embeddings1_xl_fx), normalize(embeddings1_rl_sq2_fx),
                                  normalize(embeddings1_dl_fx), normalize(embeddings1_el_s2_rmdp_fx), normalize(embeddings1_rl_fx_old),  
                                  ), axis=1)

    embeddings2 = np.concatenate((normalize(embeddings2_longformer_tqa_fx), normalize(embeddings2_bl_mlm_rmdp_fx), normalize(embeddings2_rl_sq2_mlm_only_pl_fx),
                                  normalize(embeddings2_dl_mlm_rmdp_fx), normalize(embeddings2_xl_fx), normalize(embeddings2_rl_sq2_fx),
                                  normalize(embeddings2_dl_fx), normalize(embeddings2_el_s2_rmdp_fx), normalize(embeddings2_rl_fx_old),  
                                  ), axis=1)

    embeddings3 = np.concatenate((normalize(embeddings3_longformer_tqa_fx), normalize(embeddings3_bl_mlm_rmdp_fx), normalize(embeddings3_rl_sq2_mlm_only_pl_fx),
                                  normalize(embeddings3_dl_mlm_rmdp_fx), normalize(embeddings3_xl_fx), normalize(embeddings3_rl_sq2_fx),
                                  normalize(embeddings3_dl_fx), normalize(embeddings3_el_s2_rmdp_fx), normalize(embeddings3_rl_fx_old),  
                                  ), axis=1)

    embeddings4 = np.concatenate((normalize(embeddings4_longformer_tqa_fx), normalize(embeddings4_bl_mlm_rmdp_fx), normalize(embeddings4_rl_sq2_mlm_only_pl_fx),
                                  normalize(embeddings4_dl_mlm_rmdp_fx), normalize(embeddings4_xl_fx), normalize(embeddings4_rl_sq2_fx),
                                  normalize(embeddings4_dl_fx), normalize(embeddings4_el_s2_rmdp_fx), normalize(embeddings4_rl_fx_old),  
                                  ), axis=1)

    embeddings5 = np.concatenate((normalize(embeddings5_longformer_tqa_fx), normalize(embeddings5_bl_mlm_rmdp_fx), normalize(embeddings5_rl_sq2_mlm_only_pl_fx),
                                  normalize(embeddings5_dl_mlm_rmdp_fx), normalize(embeddings5_xl_fx), normalize(embeddings5_rl_sq2_fx),
                                  normalize(embeddings5_dl_fx), normalize(embeddings5_el_s2_rmdp_fx), normalize(embeddings5_rl_fx_old),  
                                  ), axis=1)


    ### Where to load the gpr weights, embedding and target for GPR inference
    gpr_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/best_9_gpr_model_test'
    emb_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/best_9_embeddings_test.npy'
    y_path = '../input/embeddings/fx_embeddings_and_folds/fx_embeddings/gpr_rbf/best_9_y_test.npy'

    embeddings = [embeddings1, embeddings2, embeddings3, embeddings4, embeddings5]

    print("Starting GPR inference")
    gpr_preds_concat = [get_pyro_emb_preds(emb_path, y_path, gpr_path, x) for x in embeddings]

    preds_combine = np.mean(gpr_preds_concat, axis=0)
    print(preds_combine)