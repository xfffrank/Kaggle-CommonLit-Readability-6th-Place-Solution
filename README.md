# Kaggle-CommonLit-6th-Place-Solution

* Solution summary:

## Archive Contents
* To be filled...

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
1. Train the listed models in our summary by changing the configurations in `config.py` and running `train.py` in the `src` folder.
2. Train the GPR model by concatenating all of the oof embeddings.

## Inference
* Use the xxx file to run inference with the trained models.

## License
* All of our solution code are open-sourced under the Apache 2.0 license.
