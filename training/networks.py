import pytorch_lightning as pl
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AutoConfig
import torch
from torch import nn
import transformers

from config import CFG


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):  # features: (batch_sz, sequence_len, hidden_size)
        att = torch.tanh(self.W(features))  # (batch_sz, sequence_len, hidden_dim)
        score = self.V(att)  # (batch_sz, sequence_len, 1)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)  # (batch_sz, hidden_size)
        return context_vector


class CommonLitModel(pl.LightningModule):
    
    def __init__(self, train_steps, val_size):
        super().__init__()
        self.save_hyperparameters()  # save the hparams in the checkpoint
        self.train_steps = train_steps
        self.val_size = val_size
        self.config = self.get_config()
        self.backbone = self.get_backbone(self.config)
        self.num_labels = 1
        self.clf = nn.Linear(self.config.hidden_size, self.num_labels)
        if CFG.dropout > 0:
            self.dropout = nn.Dropout(CFG.dropout)
        if CFG.backbone_out == 'attention':
            self.head = AttentionHead(self.config.hidden_size, self.config.hidden_size)
        if 'large' in CFG.model_name:
            # Large models perform better when using the weight initialisation method.
            self.layer_norm = nn.LayerNorm(self.config.hidden_size)
            self._init_weights(self.clf)
            self._init_weights(self.layer_norm)

    def get_config(self):
        config = AutoConfig.from_pretrained(CFG.model_path)
        if CFG.remove_dp_in_bert:
            print('[INFO] Removing dropout in the transformer.')
            keys = vars(config).keys()
            if "hidden_dropout_prob" in keys:
                dropout_name = "hidden_dropout_prob"
            elif "dropout" in keys:
                dropout_name = "dropout"
            else:
                raise NotImplementedError()
            print(f'[INFO] Dropout name in the transformer config: {dropout_name}')
            config.update({
                dropout_name: 0.0,
                "layer_norm_eps": 1e-7,
            })
        return config

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
        model = transformers.AutoModel.from_pretrained(CFG.model_path, config=config)
        if CFG.reinit_layer == 'pooler':
            print('Reinitializing Pooler Layer ...')
            encoder_temp = model
            encoder_temp.pooler.weight.data.normal_(mean=0.0, std=config.initializer_range)
            encoder_temp.pooler.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
            print('Done.!')
        elif CFG.reinit_layer > 0:
            print(f'Reinitializing Last {CFG.reinit_layer} Layers ...')
            num_modules = 0
            if 'bart' in CFG.model_name or 'led' in CFG.model_name:
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
        elif CFG.backbone_out == 'mean_pooling':
            last_hidden_state = out['last_hidden_state']  # (batch_size, max_len, hidden_size)
            input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # (batch_size, max_len, hidden_size)
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)  # (batch_size, hidden_size)
            sum_mask = input_mask_expanded.sum(1)  # (batch_size, hidden_size)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)
            out = mean_embeddings
        elif CFG.backbone_out == 'attention':
            if 'pooler_output' in out.keys():
                pooler = out['pooler_output'].reshape((out['pooler_output'].shape[0], 1, self.config.hidden_size))
                x = torch.cat([out['last_hidden_state'], pooler],axis=1)  # (batch size, sequence len + 1, hidden size)
            else:
                x = out['last_hidden_state']
            out = self.head(x)
        return out
    
    def forward(self, inputs, return_features=False):
        out = self.extract_feats(inputs)
        if return_features:
            feats = out
        if 'large' in CFG.model_name:
            out = self.layer_norm(out)
        if CFG.dropout > 0:
            out = self.dropout(out)
        out = self.clf(out).squeeze()
        if return_features:
            return out, feats
        else:
            return out
    
    def training_step(self, batch, batch_idx):
        ids, mask, y = batch
        pred = self.forward([ids, mask])
        pred = pred.view(-1,1)
        y = y.view(-1,1)  # prevent warning about size
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids, mask, y = batch
        pred = self.forward([ids, mask])
        pred = pred.view(-1,1)
        y = y.view(-1,1)
        loss = self.loss_fn(pred, y, 'sum')
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.sqrt(torch.stack(outputs).sum() / self.val_size)
        self.log("val_loss", loss)
        
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = AdamW(params, lr=CFG.learning_rate, weight_decay=CFG.wts_decay, betas=CFG.betas)
        total_steps = CFG.epochs * self.train_steps
        warmup_steps = int(CFG.warmup_ratio * total_steps)
        print(f'[INFO] Total steps: {total_steps}')
        print(f'[INFO] Warmup steps: {warmup_steps}')
        if CFG.decay == 'constant':
            return [optimizer]
        elif CFG.decay == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        elif CFG.decay == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        elif CFG.decay == 'constant_s':
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        else:
            raise NotImplementedError()
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler]
    
    def loss_fn(self, y_pred, y_true, mode='mean'):
        """MSE"""
        criterion = nn.MSELoss(reduction=mode)
        return criterion(y_pred, y_true)
