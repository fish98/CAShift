# CAShift - Benchmark

This repo contains models used in our paper, including our self-implemented AutoEncoder (AE), Variational AutoEncoder (VAE) model with semantic-awared embeddings. Different configuration files can be used for reproducing our evaluation experiment such as RQ1: In Distribution Evaluation, RQ2: Shift Scenario Evaluation and RQ3: Continous Learning with Retrain for Shift Adaptation.

## Requrements

python=3.10

Default requirements.txt works on:

***NVIDIA-SMI 535.104.12, Driver Version: 535.104.12, CUDA Version: 12.2, Card: NVIDIA L40***

and

***NVIDIA-SMI 515.65.01, Driver Version: 515.65.01, CUDA Version: 11.7, Card: NVIDIA RTX A5000***

## Experiment

<!-- Check whether output dir exist before running `run_train.py` -->

### Construct embedding

python data_processing.py

The output should be the feature directory containing .pt files (e.g, embedding.pt) for the next step.

### Training and testing

python main.py --config=configs/baseline.yaml (train and test)

python test_only.py (test only)

python main.py --config=configs/retrain.yaml (only for retrain and test)

AE.pth and VAE.pth trained model output

### Debugging

The code has been modified to remove sensitive directory names due to privacy concerns. As a result, there may be some bugs that require further debugging...