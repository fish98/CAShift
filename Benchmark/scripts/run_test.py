import torch
from transformers import BertTokenizer, BertModel
from dataset import LogAEDataset
from model import AE
from tqdm import tqdm
import numpy as np
from sklearn import metrics

device = torch.device("cuda")
model_path = "./model.pth"
state_dict = torch.load(model_path)
model = AE().to(device)
model.load_state_dict(state_dict)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)
bert.eval()

anomaly_data_dir = '/data/c/visitor/CAShift-CSV/test-attack'
normal_data_dir = '/data/c/visitor/CAShift-CSV/test-normal'

anomaly_dataset = LogAEDataset(anomaly_data_dir, tokenizer, bert, embedding_file='./test_anomaly_embeddings.pt')
normal_dataset = LogAEDataset(normal_data_dir, tokenizer, bert, embedding_file='./test_normal_embeddings.pt')

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

print("Anomaly scores: ", sum(anomaly_scores)/len(anomaly_scores))
print("Normal scores: ", sum(normal_scores)/len(normal_scores))
pass

merged_results = [(score, 1) for score in anomaly_scores] + [(score, 0) for score in normal_scores]
y_pred = [score for score, _ in merged_results]
y_true = [label for _, label in merged_results]

auc = metrics.roc_auc_score(y_true, y_pred)
precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-50)
max_index = np.argmax(f_scores[:-1])  # Drop the last f-score which has no corresponding threshold.
precision, recall, threshold, f_score = precisions[max_index], recalls[max_index], thresholds[max_index], f_scores[max_index]
print(f"AUC: {auc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f_score}")