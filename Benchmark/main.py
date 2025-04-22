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
from test_auto import batch_testing

attacks = [
    'CVE-2015-8562', 'CVE-2016-4029', 'CVE-2017-8917', 'CVE-2019-5736',
    'CVE-2020-14386', 'CVE-2021-23132', 'CVE-2021-25743', 'CVE-2022-1708',
    'CVE-2024-21626', 'CVE-2016-10033', 'CVE-2016-5487', 'CVE-2019-17671',
    'CVE-2019-8341', 'CVE-2020-15257', 'CVE-2021-25742', 'CVE-2021-30465',
    'CVE-2023-23752', 'CWE-200', 'CVE-2024-1086', 'CWE-400'
]

shifts = ['App-1', 'Arch-1', 'Version-1'] + ['App-2', 'Arch-2', 'Version-2']


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


def eval_metrics(c, logger, y_true, y_pred, spec='base'):
    auc = metrics.roc_auc_score(y_true, y_pred)
    if auc < 0.5:
        y_pred = [-score for score in y_pred]
        print("Inverting scores")
        auc = metrics.roc_auc_score(y_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-50)
    max_index = np.argmax(f_scores[:-1])  # Drop the last f-score which has no corresponding threshold.
    precision, recall, threshold, f_score = precisions[max_index], recalls[max_index], thresholds[max_index], f_scores[max_index]

    logger.info(f"Evaluating under {spec} setting")
    if spec == 'normal+shift vs anomaly':
        logger.info("\n=============================================")
        logger.info(f"AUC: {auc}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f_score}")
        logger.info(f"Threshold: {threshold}")
        logger.info("=============================================\n")
    else:
        logger.info(f"AUC: {auc}, Precision: {precision}, Recall: {recall}, F1 Score: {f_score}, Threshold: {threshold}")


def get_exp_name(c):
    exp_name = f"{c.shift_spec}_{c.attack_spec}_{c.model}"
    if c.get('use_template', False):
        exp_name += "_template"
    return exp_name


def data_processor(c, tokenizer, bert):
    use_template = c.get('use_template', False)
    truncate_at = c.get('truncate_at', 50)
    if use_template:
        c.embedding_file = c.embedding_file.replace('.pt', '_template.pt')
        c.embedding_test_normal = c.embedding_test_normal.replace('.pt', '_template.pt')
        c.embedding_test_anomaly = c.embedding_test_anomaly.replace('.pt', '_template.pt')

    # Process training data
    preprocess_data(c.data_dir, tokenizer, bert, c.feature_dir, c.embedding_file, c.window_size, use_template, truncate_at)

    # Process testing data
    c.test_feature_dir = c.get('test_feature_dir', '') or c.feature_dir

    # # process test-normal
    preprocess_data(c.test_normal_dir, tokenizer, bert, c.test_feature_dir, c.embedding_test_normal, c.window_size, use_template, truncate_at, save_every_k=-1)

    # # process test-anomaly
    if 'ATTACKHOLDER' in c.test_anomaly_dir:
        c.test_anomaly_dir = c.test_anomaly_dir.replace('ATTACKHOLDER', c.attack_spec)
        c.embedding_test_anomaly += f"_{c.attack_spec}"
    preprocess_data(c.test_anomaly_dir, tokenizer, bert, c.test_feature_dir, c.embedding_test_anomaly, c.window_size, use_template, truncate_at, save_every_k=-1)

    if c.get('embedding_test_shift', None):
        # # process test-shift
        c.test_shift_dir = c.test_shift_dir.replace('SHIFTHOLDER', c.shift_spec)
        c.embedding_test_shift += f"_{c.shift_spec}"
        preprocess_data(c.test_shift_dir, tokenizer, bert, c.test_feature_dir, c.embedding_test_shift, c.window_size, use_template, truncate_at, save_every_k=-1)

    return c


def train(c, test_only=False):
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

    # Data preprocessing
    if c.attack_spec == 'all' or c.shift_spec == 'all':
        logger.warning("Assuming data is already preprocessed. When setting attack_spec or shift_spec to 'all', ensure that the data is preprocessed.")
    else:
        c = data_processor(c, tokenizer, bert)

    if test_only:
        assert os.path.exists(model_path), f"Model file {model_path} does not exist"
        state_dict = torch.load(model_path)
        model = eval(c.model)().to(accelerator.device)
        model.load_state_dict(state_dict)
        model.eval()

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
        dataset = LogAEDataset(c.feature_dir, c.embedding_file, accelerator.device)

        # Resume & Retrain
        if c.get('resume', None):
            resume_model_path = os.path.join(c.resume, f'{c.model}.pth')
            state_dict = torch.load(resume_model_path)
            logger.info(f"Resuming from {resume_model_path}")
            model.load_state_dict(state_dict)
            # Apply retrain data filtering
            dataset.apply_filtering(c)

        data_loader = DataLoader(dataset, batch_size=c.bsz, shuffle=True)

        num_epoch = c.num_epoch
        progress_bar = tqdm(range(num_epoch * len(data_loader)), desc="Trainning")

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

    # Testing
    atts = [c.attack_spec] if c.attack_spec != 'all' else attacks
    shs = [c.shift_spec] if c.shift_spec != 'all' else shifts

    batch_testing(c.test_feature_dir, c.exp_dir, atts, shs, c.model)


def batch_exp_wrapper(attacks, shifts, base_config, test_only):
    i = 0
    for attack in attacks:
        for shift in shifts:
            if i > 0:
                test_only = True
            print(f"Running experiment with attack: {attack} and shift: {shift}")
            config = OmegaConf.create(base_config)
            config.attack_spec = attack
            config.shift_spec = shift
            train(config, test_only)
            i += 1


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/retrain.yaml")
    # parser.add_argument("-c", "--config", type=str, default="configs/shift.yaml")
    # parser.add_argument("-c", "--config", type=str, default="configs/template.yaml")
    parser.add_argument("-t", "--test-only", type=str2bool, default='no')
    parser.add_argument("-b", "--batch-run", type=str2bool, default='no')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.batch_run:
        batch_exp_wrapper(attacks, shifts, config, args.test_only)
    else:
        train(config, args.test_only)
