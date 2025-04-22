import torch
import os
import random

class LogAEDataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, embedding_file, device='cuda'):
        embedding_file_path = os.path.join(feature_dir, embedding_file) + '.pt'
        assert os.path.exists(embedding_file_path), 'Embedding file does not exist'
        self.embeddings = torch.load(embedding_file_path)
        self.device = device

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx].to(self.device)

    def get_embeddings(self):
        return self.embeddings

    def custom_filtering_func(self, full_list_sorted, percentage, head_or_tail, score_dict):
        # head use those with smaller losses; tail use those with larger losses
        if head_or_tail == 'zol':# head
            return full_list_sorted[:int(len(full_list_sorted) * percentage)]
        elif head_or_tail == 'gini': # tail
            return full_list_sorted[int(len(full_list_sorted) * (1 - percentage)):]
        elif head_or_tail == 'kmst':
            # find the highest score and lowest score
            output_index = []
            max_score = max(score_dict.values())
            min_score = min(score_dict.values())
            # preset seperate number = 5
            seperate_num = 5
            stp = (max_score - min_score) / 5
            thd = [min_score + i * stp for i in range(1, 5)]
            seperate_score = {i: [] for i in range(5)} # store all seperate score
            for index, value in score_dict.items():
                if value <= thd[0]:
                    seperate_score[0].append(index)
                elif value <= thd[1]:
                    seperate_score[1].append(index)
                elif value <= thd[2]:
                    seperate_score[2].append(index)
                elif value <= thd[3]:
                    seperate_score[3].append(index)
                else:
                    seperate_score[4].append(index)
            # random select k samples from each seperate score
            for item in seperate_score.values():
                k = int(max(1, len(item) * percentage))
                if len(item) > 0:
                    select_index = random.sample(item, k)
                    output_index.extend(select_index)
            return output_index
        else:
            print("WARNNING: Unknown filtering method!")
            return full_list_sorted[int(len(full_list_sorted) * (1 - percentage)):]

    def apply_filtering(self, c):
        score_file_path = c.score_file.replace('HOLDER', c.model)
        rst_dct = torch.load(score_file_path)
        test_exp_specs = rst_dct.keys()
        demo_spec = list(test_exp_specs)[0]
        scores = rst_dct[demo_spec]['normal']
        score_dict = {i: scores[i] for i in range(len(scores))}
        sorted_keys = sorted(score_dict, key=score_dict.get)
        filtered_indices = self.custom_filtering_func(sorted_keys, c.percentage, c.head_or_tail, score_dict)
        self.embeddings = [self.embeddings[i] for i in filtered_indices]
