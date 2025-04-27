import torch
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
from transformers import *

class DatasetMetaQA(Dataset):
    def __init__(self, data, en_emb_dict, re_emb_dict, entity2idx, relation2idx,
                 entity_dim, node_rels_map):
        self.data = data
        self.en_emb_dict = en_emb_dict
        self.re_emb_dict = re_emb_dict
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.en_emb_dict.keys())
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.relation2idx = relation2idx
        self.entity_dim = entity_dim
        self.node_rels_map = node_rels_map
        self.err_count = 0

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr
    
    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def relToOneHot2(self, indices):
        num_rel = len(indices)
        indices = torch.LongTensor(indices)
        vec_len = len(self.relation2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_text = question_text #+ " " + data_point[-1]

        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head = data_point[0]

        rel_ids = []
        
        if isinstance(data_point[3], str):
            rel_name = data_point[3].strip()
            if rel_name in self.re_emb_dict:
                rel_ids.append(self.re_emb_dict[rel_name])
            
        else:
            for rel_name in data_point[3]:
                if rel_name in self.re_emb_dict:
                    rel_ids.append(self.re_emb_dict[rel_name])
        path_emb = rel_ids
        
        path_en_embs = []
        for en in data_point[4]:
            if en in self.en_emb_dict:
                path_en_embs.append(self.en_emb_dict[en])
            
        en_tensors = [v.clone().detach() for v in path_en_embs]
        path_en_embs = torch.stack(en_tensors, dim=0)

        neighbors = self.node_rels_map[data_point[0].strip()]

        neighbor_ids = []
        for n in neighbors:
            if n in self.relation2idx:
                neighbor_ids.append(self.relation2idx[n])
        neighbor_rel_onehot = self.relToOneHot2(neighbor_ids)

        if question_tokenized.shape[0] != 64:
            self.err_count += 1
            return
        
        return question_tokenized, attention_mask, head, path_emb, path_en_embs, neighbor_rel_onehot


    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)

        if len(question_tokenized) != 64:
            print("question_tokenized = ", len(question_tokenized))

        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

