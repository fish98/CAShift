import torch
from transformers import BertTokenizer, BertModel
from dataset import LogAEDataset
from model import AE, VAE
from tqdm import tqdm
import numpy as np
from sklearn import metrics
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


def eval_metrics(logger, y_true, y_pred, threshold=-1, spec='base'):
    auc = metrics.roc_auc_score(y_true, y_pred)
    if auc < 0.5:
        y_pred = [-score for score in y_pred]
        print("Inverting scores")
        auc = metrics.roc_auc_score(y_true, y_pred)
        threshold = -threshold if threshold != -1 else threshold

    if threshold == -1:
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-50)
        max_index = np.argmax(f_scores[:-1])  # Drop the last f-score which has no corresponding threshold.
        precision, recall, threshold, f_score = precisions[max_index], recalls[max_index], thresholds[max_index], f_scores[max_index]
    else:
        precision = metrics.precision_score(y_true, [1 if score > threshold else 0 for score in y_pred])
        recall = metrics.recall_score(y_true, [1 if score > threshold else 0 for score in y_pred])
        f_score = metrics.f1_score(y_true, [1 if score > threshold else 0 for score in y_pred])

    # Change calculation algorithm
    if threshold < 0:
        pre_recall = recall
        recall = 1 - recall
        precision = recall * 100 / ((100 - 100 * pre_recall/precision + 1e-50) + 100 * recall)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-50)

    logger.info(f"\n{spec}")
    logger.info(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f_score:.4f}")

    return auc, precision, recall, f_score, threshold


def testing(normal_dataset, anomaly_dataset, model, logger, threshold=-1, spec='normal vs anomaly'):
    normal_scores = []
    anomaly_scores = []

    for data in tqdm(normal_dataset, desc="Evaluating Normal"):
        data = data.unsqueeze(0)
        with torch.no_grad():
            score = model(data)
        normal_scores.append(score.item())

    for data in tqdm(anomaly_dataset, desc="Evaluating Anomaly"):
        data = data.unsqueeze(0)
        with torch.no_grad():
            score = model(data)
        anomaly_scores.append(score.item())

    merged_results = [(score, 0) for score in normal_scores] + [(score, 1) for score in anomaly_scores]
    y_pred = [score for score, _ in merged_results]
    y_true = [label for _, label in merged_results]
    auc, precision, recall, f_score, threshold = eval_metrics(logger, y_true, y_pred, threshold=threshold, spec=spec)

    logger.info(f"Average normal scores: {sum(normal_scores) / len(normal_scores)}")
    logger.info(f"Average anomaly scores: {sum(anomaly_scores) / len(anomaly_scores)}")
    logger.info(f"Threshold: {threshold}")

    sub_rst_dict = {
        'normal': normal_scores,
        'anomaly': anomaly_scores,
        'threshold': threshold,
        'AUC': auc,
        'precision': precision,
        'recall': recall,
        'f_score': f_score
    }

    return threshold, sub_rst_dict


def batch_testing(test_feature_dir,
                  exp_dir,
                  attacks,
                  shifts,
                  model_type):
    os.makedirs(exp_dir, exist_ok=True)

    att_type = 'aggr' if len(attacks) == 1 and attacks[0] == 'aggregated' else 'split'
    logger = get_logger(exp_dir, f'{att_type}_test_{model_type}')

    # dynamic model loading
    model = eval(model_type)().cuda()
    model_path = os.path.join(exp_dir, f'{model_type}.pth')

    assert os.path.exists(model_path), f"Model file {model_path} does not exist"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    embedding_test_normal = f"test_normal_embeddings"
    normal_dataset = LogAEDataset(test_feature_dir, embedding_test_normal)
    # embedding_test_normal = f"retrain_shift_embeddings_App-1"   # replace here #########
    # retrain_feature_dir = '/CAShift/Benchmark/RetrainFeature'
    # normal_dataset = LogAEDataset(retrain_feature_dir, embedding_test_normal)

    result_dict = {}

    # Testing
    for att in attacks:
        embedding_test_anomaly = f"test_anomaly_embeddings_{att}"
        anomaly_dataset = LogAEDataset(test_feature_dir, embedding_test_anomaly)

        threshold, sub_rst_dict = testing(normal_dataset, anomaly_dataset, model, logger, spec=f'normal vs {att}')

        result_dict[f'normal+{att}'] = sub_rst_dict

        for sh in shifts:
            embedding_test_shift = f"test_shift_embeddings_{sh}"
            shift_dataset = LogAEDataset(test_feature_dir, embedding_test_shift)

            _, sub_rst_dict = testing(shift_dataset, anomaly_dataset, model, logger, threshold=threshold, spec=f'{sh} vs {att}')
            result_dict[f'{sh}+{att}'] = sub_rst_dict

    torch.save(result_dict, os.path.join(exp_dir, f'result_dict_{model_type}.pth'))


def str2bool(v):
    return v.lower() in ("y", "yes", "true", "t", "1")


if __name__ == '__main__':
    # attacks = [
    #     'CVE-2015-8562', 'CVE-2016-4029', 'CVE-2017-8917', 'CVE-2019-5736',
    #     'CVE-2020-14386', 'CVE-2021-23132', 'CVE-2021-25743', 'CVE-2022-1708',
    #     'CVE-2024-21626', 'CVE-2016-10033', 'CVE-2016-5487', 'CVE-2019-17671',
    #     'CVE-2019-8341', 'CVE-2020-15257', 'CVE-2021-25742', 'CVE-2021-30465',
    #     'CVE-2023-23752', 'CWE-200', 'CVE-2024-1086', 'CWE-400'
    # ]

    # shifts = ['App-1', 'Arch-1', 'Version-1'] + ['App-2', 'Arch-2', 'Version-2']

    shifts = []
    attacks = ['aggregated']

    batch_testing('/CAShift/Benchmark/FinalFeature',
                  # '/CAShift/Benchmark/exps/v2/aggr',
                #   '/CAShift/Benchmark/exps/demo_retrain',
                "/CAShift/Benchmark/exps/test_retrain",
                  attacks,
                  shifts,
                  'AE')
