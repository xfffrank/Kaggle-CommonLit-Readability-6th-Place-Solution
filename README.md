# Kaggle-CommonLit-6th-Place-Solution

* Solution summary: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258554

## Model overview of our final submission

| Model Name | CV | Public | Private | notes
| --- | --- | --- | --- | --- |
| roberta-large | 0.492 | 0.471 | 0.471| mlm on training set
| deberta-large | 0.485 |0.474 | 0.476| mlm on training set
| xlnet-large-cased | 0.494 | 0.475 | 0.476| 
| deepset/roberta-large-squad2 | 0.488 | 0.464 | 0.467| 
| deepset/roberta-large-squad2 | 0.484 | 0.466 | 0.464 | mlm on train set and external data
| allenai/longformer-large-4096-finetuned-triviaqa | 0.489 | 0.467 | 0.47 | 
| valhalla/bart-large-finetuned-squadv1 | 0.471 | 0.462 | 0.466 |  mlm on train set and external data, remove dropout
| microsoft/deberta-large-mnli | 0.469 | 0.462 | 0.469 | mlm on train set and external data, remove dropout
| ahotrod/electra_large_discriminator_squad2_512 | 0.477 | 0.468 | 0.468 | remove dropout

**Notes**
* "mlm" refers to Masked Language Modelling pretraining.
* To remove dropout, set `remove_dp_in_bert = True` in the `config.py`.

## Archive Contents
* `training` folder: code for training transformer models.
* `inference` folder: training and inference code for the GPR model
    - `GPR_base.py`: script needed to train the GPR model
    - `GPR_inference.py`: demonstration of our final submission
* `external_data` folder: external data for the use of pretraining. The data come from the same url as in `url_legal` of the original `train.csv` and we only used those with CC-BY and CC-BY-SA licenses.

## Hardware
* RTX 3090 * 2

## Software
The main packages used are listed below. Please see the `requirements.txt` for the complete list.
* pytorch-lightning==1.3.1
* torch==1.8.1+cu111
* pyro-api==0.1.2
* pyro-ppl==1.6.0
* pandas==1.2.4
* numpy==1.19.5
* scikit-learn==0.24.2
* scipy==1.6.3

## Data Setup
* Please add the `train.csv` file from https://www.kaggle.com/c/commonlitreadabilityprize/data as the input.

## Training
1. Train the listed models in our summary by changing the configurations in `config.py` and running `train.py` in the `training` folder.
    - e.g. To train a roberta-large model, change `model_path` and `model_name` in  `config.py` to `roberta-large` and run `train.py`. Use the same `model_path` and `model_name` as in the "Model Name" column above if no pretraining is needed.
    - For models with mlm pretraining needed, use the original `train.csv` (combined with the provided external data) and run `pretrain.py` to get the pretrained model first. During finetuning, set `model_path` in `config.py` to the pretrained model path.
2. Train a GPR model by concatenating all of the out-of-fold embeddings for the 9 models.
    - The script needed is `GPR_base.py` in the `inference` folder.
    - Suppose the dimension of embeddings is 1024, and the number of samples is 2834 and the number of models is 9, you will get the concatenated embedding of the size (2834, 9*1024) for training the GPR.

## Inference
* Use `GPR_inference.py` in the `inference` folder to run inference with the trained models.

## License
* All of our solution code are open-sourced under the Apache 2.0 license.
