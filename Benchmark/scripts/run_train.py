import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import BertTokenizer, BertModel
from dataset import LogAEDataset
from model import AE
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from omegaconf import OmegaConf
import argparse
from data_processing import preprocess_data
import os


def train(c, test_only=False):
    os.makedirs(c.exp_dir, exist_ok=True)
    accelerator = Accelerator()

    tokenizer = BertTokenizer.from_pretrained(c.tokenizer)
    bert = BertModel.from_pretrained(c.bert_model).to(accelerator.device)
    bert.eval()

    model = AE().to(accelerator.device)
    model_path = os.path.join(c.exp_dir, 'model.pth')

    # Data preprocessing
    preprocess_data(c.data_dir, tokenizer, bert, c.feature_dir, c.embedding_file)
    preprocess_data(c.test_normal_dir, tokenizer, bert, c.feature_dir, c.embedding_test_normal)
    preprocess_data(c.test_anomaly_dir, tokenizer, bert, c.feature_dir, c.embedding_test_anomaly)

    if not test_only:
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
    else:
        assert os.path.exists(model_path), f"Model file {model_path} does not exist"
        state_dict = torch.load(model_path)
        model = AE().to(accelerator.device)
        model.load_state_dict(state_dict)
        model.eval()

    # Testing
    anomaly_dataset = LogAEDataset(c.feature_dir, c.embedding_test_anomaly, accelerator.device)
    normal_dataset = LogAEDataset(c.feature_dir, c.embedding_test_normal, accelerator.device)

    anomaly_scores = []
    normal_scores = []

    for data in tqdm(anomaly_dataset, desc="Evaluating Anomaly"):
        data = data.unsqueeze(0)
        with torch.no_grad():
            score = model(data)
        anomaly_scores.append(score.item())

    for data in tqdm(normal_dataset, desc="Evaluating Normal"):
        data = data.unsqueeze(0)
        with torch.no_grad():
            score = model(data)
        normal_scores.append(score.item())

    print("Average anomaly scores: ", sum(anomaly_scores) / len(anomaly_scores))
    print("Average normal scores: ", sum(normal_scores) / len(normal_scores))

    merged_results = [(score, 1) for score in anomaly_scores] + [(score, 0) for score in normal_scores]
    y_pred = [score for score, _ in merged_results]
    y_true = [label for _, label in merged_results]

    auc = metrics.roc_auc_score(y_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-50)
    max_index = np.argmax(f_scores[:-1])  # Drop the last f-score which has no corresponding threshold.
    precision, recall, threshold, f_score = precisions[max_index], recalls[max_index], thresholds[max_index], f_scores[
        max_index]
    print(f"AUC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f_score}")
    print(f"Threshold: {threshold}")

    with open(os.path.join(c.exp_dir, "results.txt"), mode="w") as f:
        f.write(f"AUC: {auc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f_score}\n")
        f.write(f"Threshold: {threshold}\n")


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("-t", "--test-only", type=str2bool, default='no')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    train(config, args.test_only)
