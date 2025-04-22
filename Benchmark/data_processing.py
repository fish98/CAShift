import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from omegaconf import OmegaConf
import re

def extract_number(file_name):
    match = re.search(r'normal_(\d+)\.csv', file_name)
    return int(match.group(1)) if match else -1



def encode_chunks(calls, tokenizer, bert, chunk_size, batch_size=512):
    all_inputs = []
    for i in range(0, len(calls), chunk_size):
        chunk = calls[i:i + chunk_size]
        text = ' '.join(chunk)
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).to(bert.device)
        all_inputs.append(input_ids)

    all_inputs = pad_sequence(all_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    all_masks = (all_inputs != tokenizer.pad_token_id).long()

    cls_states = []
    with torch.no_grad():
        for i in range(0, len(all_inputs), batch_size):
            input_batch = all_inputs[i:i + batch_size]
            mask_batch = all_masks[i:i + batch_size]
            last_hidden_states = bert(input_batch, attention_mask=mask_batch)[0]
            cls_state = last_hidden_states[:, 0, :]
            cls_states.append(cls_state)
    cls_state = torch.vstack(cls_states)
    return cls_state


def preprocess_data(data_dir, tokenizer, bert, feature_dir, output_file_name, chunk_size,
                    use_template=False, truncate_at=50, save_every_k=-1):
    os.makedirs(feature_dir, exist_ok=True)
    output_file = os.path.join(feature_dir, output_file_name) + '.pt'
    if os.path.exists(output_file):
        print(f'Embeddings already exist at {output_file}. Skipping...')
        return
    files = sorted(os.listdir(data_dir), key=extract_number)
    all_embeddings = []
    for i, file in enumerate(tqdm(files, desc='Processing files')):
        if not file.endswith('.csv'):
            continue

        if save_every_k != -1 and os.path.exists(os.path.join(feature_dir, f"{output_file_name}_{i // save_every_k}")):
            continue

        df = pd.read_csv(os.path.join(data_dir, file))
        calls = df['SysCall'].tolist() if not use_template else df['EventTemplate'].tolist()
        calls = [call[:truncate_at] for call in calls]
        file_embedding = encode_chunks(calls, tokenizer, bert, chunk_size=chunk_size)
        file_embedding = file_embedding.cpu()
        all_embeddings.append(file_embedding)
        if save_every_k != -1 and (i + 1) % save_every_k == 0:
            output_file = os.path.join(feature_dir, f"{output_file_name}_{i // save_every_k}") + '.pt'
            torch.save(all_embeddings, output_file)
            print(f'Saved embeddings to {output_file}')
            all_embeddings = []
    if save_every_k == -1:
        torch.save(all_embeddings, output_file)
        print(f'Saved embeddings to {output_file}')


if __name__ == '__main__':
    c = OmegaConf.load('configs/shift.yaml')
    device = torch.device('cuda:1')
    tokenizer = BertTokenizer.from_pretrained(c.tokenizer)
    bert = BertModel.from_pretrained(c.bert_model).to(device)
    bert.eval()
    
    # for training data
    preprocess_data(c.data_dir, tokenizer, bert, c.feature_dir, c.embedding_file, c.window_size, use_template=c.use_template)
    
    # for shift data
    ets_base = c.embedding_test_shift
    for sh in ['App-1', 'Arch-1', 'Version-1', 'App-2', 'Arch-2', 'Version-2']:
        c.test_shift_dir = c.test_shift_dir.replace('SHIFTHOLDER', sh)
        c.embedding_test_shift = ets_base + f"_{sh}"
        preprocess_data(c.test_shift_dir, tokenizer, bert, c.feature_dir, c.embedding_test_shift, c.window_size)
    
    # for attack data
    # preprocess_data(c.test_anomaly_dir, tokenizer, bert, c.feature_dir, c.embedding_test_anomaly, c.window_size, use_template=c.use_template)
    # preprocess_data('/CAShift-Dataset/Test-Attack/CVE-2024-1086', tokenizer, bert, './feat/', 'test_anomaly_embeddings_CVE-2024-1086', c.window_size)
    preprocess_data('/CAShift-Dataset/Test-Attack/aggregated', tokenizer, bert, c.feature_dir, 'test_anomaly_embeddings_aggregated', c.window_size)
