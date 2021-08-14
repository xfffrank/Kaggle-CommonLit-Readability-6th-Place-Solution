# Kaggle-CommonLit-6th-Place-Solution

* Solution summary: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258554
* Notebook link: https://www.kaggle.com/xyzdivergence/gpr-final

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
    - `pretrain.py`: MLM pretraining script.
    - `train.py`: Finetuning script.
* `inference` folder: training and inference code for the GPR model
    - `GPR_training.py`: GPR training script
    - `GPR_inference.py`: Inference with GPR for our final submission
* `external_data` folder: external data for the use of pretraining. The data come from the same url as in `url_legal` of the original `train.csv` and we only used those with CC-BY and CC-BY-SA licenses. Specifically, we scraped the texts from the original URLs and used the same IDs in the csv file. We’ve also scraped some texts from “https://kids.frontiersin.org” which do not correspond to any ID in the training file, so the ID column is empty.

## Hardware
* RTX 3090 * 2

## Software
Please see the `requirements.txt` for the dependencies.

## Environment setup
```
$ conda create --name commonlit python=3.8
$ conda activate commonlit
$ pip install -r requirements.txt
```

* Note that if you are using GPU, you need to install the matched torch version. e.g. If you are using cuda 11.1, you need to install `torch` with
```
$ pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Configuration Setup
* Fill the "set up" section in `config.py`.

## Training
1. Train the models in the [model overview](#Model-overview-of-our-final-submission).
    1. Pre-train models listed in `pretrain_config.yaml`.  
        e.g. For the roberta-large model, run
        ```
        $ python pretrain.py --config roberta-large
        ```
        Remember to fill the `output_dir` where the checkpoint will be saved.
    2. Finetune models listed in `finetune_config.yaml`.   
        e.g. For the roberta-large-mlm model, run
        ```
        $ python train.py --config roberta-large-mlm
        ```
        For models finetuned on a pretrained checkpoint, remember to replace the `model_path` with the directory that contains the checkpoint.

2. Train a GPR model by concatenating all of the out-of-fold (OOF) embeddings for the 9 models.
    - Suppose the dimension of the embeddings for a single large model is 1024, and the number of samples in the training data is 2834, and total number of models is 9, you will get the concatenated embedding of the size (2834, 9*1024) as inputs for training the GPR model.
    - Inside the inference folder there is a `gpr_config.py` config file, change the following variables to the file location for saving the GPR model and the relevant files once the training is completed (These path should be the same for the inference as well).
      ```python
      class GPR_CFG:
          models_dir = '../../models' # the directory of all the models checkpoint (MUST EXIST)
          gpr_path = f'{models_dir}/gpr_rbf/best_9_all_y_gpr' # the path to store or load the trained GPR model on the 9 concatenated embeddings
          emb_path = f'{models_dir}/gpr_rbf/best_9_all_y_embeddings.npy' # the path to store or load the 9 concatenated     embeddings
          y_path = f'{models_dir}/gpr_rbf/best_9_all_y_y_new.npy' # the path to store or load the target label
      ```
    - The script needed for training a GPR model is `GPR_training.py` in the `inference` folder.
      ```
      $ python GPR_training.py
      ```
      Once training is finished, the GPR model, embeddings file, target label will be saved to `gpr_path`, `emb_path`, and `y_path` specified in `GPR_CFG` class.

## Inference
* Use `GPR_inference.py` in the `inference` folder to run inference with the GPR model.
* Inside the inference folder there is a `gpr_config.py` config file, change the following variables to your folder/file location.
    ```python
    class GPR_CFG:
        models_dir = '../../models' # the directory of all the models checkpoint (MUST EXIST)
        COMPUTE_CV = False # Set to True if running inference on train.csv, set to False if running inference on test.csv
        train_csv = '../../commonlitreadabilityprize/train.csv' # original train.csv (MUST EXIST)
        test_csv = '../../commonlitreadabilityprize/test.csv' # original test.csv (MUST EXIST)
        output_csv = '../../commonlitreadabilityprize/final.csv' # the path to store the generated output file with readability predictions after inference
        gpr_path = f'{models_dir}/gpr_rbf/best_9_all_y_gpr' # the path to load the trained GPR model on the 9 concatenated embeddings
        emb_path = f'{models_dir}/gpr_rbf/best_9_all_y_embeddings.npy' # the path to load the 9 concatenated embeddings
        y_path = f'{models_dir}/gpr_rbf/best_9_all_y_y_new.npy' # the path to load the target label
    ```
* Inside the inference folder run the following GPR inference to return the predictions result in a `final.csv` file with a prediction column appended to the either the original train.csv or test.csv depending if `COMPUTE_CV` is set to True (Use `train.csv`) or False (Use `test.csv`).
    ```
    $ python GPR_inference.py
    ```
* Once inference is done, the output file `final.csv` with the final predictions will be generated in the location you specified in the variable `output_csv` in `GPR_CFG` class.

## License
* All of our solution code is open-sourced under the MIT license as stated in the [competition rules](https://www.kaggle.com/c/commonlitreadabilityprize/rules).
