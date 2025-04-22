import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import BertTokenizer, BertModel
from dataset import LogAEDataset
from model import AE, VAE
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from omegaconf import OmegaConf
import argparse
from data_processing import preprocess_data
import os
import logging


def get_logger(exp_dir, exp_name):
    filename = os.path.abspath(os.path.join(exp_dir, f'{exp_name}.log'))

    # Create a custom logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(filename, mode='w')
    stream_handler = logging.StreamHandler()

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_exp_name(c):
    exp_name = f"{c.shift_spec}_{c.attack_spec}_{c.model}"
    if c.get('use_template', False):
        exp_name += "_template"
    return exp_name


def train(c):
    os.makedirs(c.exp_dir, exist_ok=True)
    accelerator = Accelerator()

    exp_name = get_exp_name(c)
    logger = get_logger(c.exp_dir, exp_name)

    tokenizer = BertTokenizer.from_pretrained(c.tokenizer)
    bert = BertModel.from_pretrained(c.bert_model).to(accelerator.device)
    bert.eval()

    # dynamic model loading
    model = eval(c.model)().to(accelerator.device)
    model_path = os.path.join(c.exp_dir, f'{c.model}.pth')

    if c.resume:
        state_dict = torch.load(c.resume)
        logger.info(f"Resuming from {c.resume}")
        model.load_state_dict(state_dict)


    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    dataset = LogAEDataset(c.feature_dir, c.embedding_file, accelerator.device)
    data_loader = DataLoader(dataset, batch_size=c.bsz, shuffle=True)

    num_epoch = c.num_epoch
    progress_bar = tqdm(range(num_epoch * len(data_loader)))

    # Training
    for epoch in range(num_epoch):
        for data in data_loader:
            loss = model(data)
            logs = {"step_loss": loss.item()}
            progress_bar.set_postfix(**logs)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

    torch.save(model.state_dict(), model_path)


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("-c", "--config", type=str, default="configs/shift.yaml")
    # parser.add_argument("-c", "--config", type=str, default="configs/template.yaml")
    parser.add_argument("-t", "--test-only", type=str2bool, default='no')
    parser.add_argument("-b", "--batch-run", type=str2bool, default='no')
    args = parser.parse_args()


    attacks = [
        'CVE-2015-8562', 'CVE-2016-4029', 'CVE-2017-8917', 'CVE-2019-5736',
        'CVE-2020-14386', 'CVE-2021-23132', 'CVE-2021-25743', 'CVE-2022-1708',
        'CVE-2024-21626', 'CVE-2016-10033', 'CVE-2016-5487', 'CVE-2019-17671',
        'CVE-2019-8341', 'CVE-2020-15257', 'CVE-2021-25742', 'CVE-2021-30465',
        'CVE-2023-23752', 'CWE-200', 'CVE-2024-1086', 'CWE-400'
    ]

    shifts = ['App-1', 'Arch-1', 'Version-1'] + ['App-2', 'Arch-2', 'Version-2']

    config = OmegaConf.load(args.config)

    if args.batch_run:
        batch_exp_wrapper(attacks, shifts, config, args.test_only)
    else:
        train(config, args.test_only)
