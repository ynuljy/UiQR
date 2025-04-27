# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
# from sklearn.feature_extraction.text import CountVectorizer
import utils
import torch

entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
edge2entities = []  # each row in edge2entities is the two entities connected by this edge
edge2relation = []  # each row in edge2relation is the relation type of this edge

e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity

def read_entities(file_name):
    d = {}
    file = open(file_name, encoding='utf-8')
    for line in file:
        name, idx = line.strip().split('\t')
        d[name] = idx
    file.close()

    return d


def loader_paths(file_name):
    data = pd.read_json(file_name)
    paths = []
    
    path_result_array = data['path_result_array']

    for triplet in path_result_array:
        if len(triplet) == 0:
            continue
        
        head, relation, tail = triplet[0]['head'], triplet[0]['relation'], triplet[0]['tail']
        paths.append((head, relation, tail))

def read_relations(file_name):
    d = {}
    file = open(file_name, encoding='utf-8')
    for line in file:
        name, index = line.strip().split('\t')
        d[name] = index

    return d

def read_triplets(triple_idx_file, triple_file, entity_dict):
    data = []
    
    with open(triple_file, 'r') as file:
        for line in file:
            head_idx, tail_idx, relation_idx = line.strip().split('\t')
            data.append((head_idx, tail_idx, relation_idx))
    file.close()

    print("===reading triple finished===")
    return data


def build_kg(train_data):
    for idx, triplet in enumerate(train_data):
        head_idx, relation_idx, tail_idx = triplet

        entity2edge_set[head_idx].add(relation_idx)
        entity2edge_set[tail_idx].add(relation_idx)
        edge2entities.append([head_idx, tail_idx])


def get_h2t(train_triplets, valid_triplets, test_triplets):
    head2tails = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        head2tails[head].add(tail)
    return head2tails


def read_embedding(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        en_id, embedding = line.strip().split('\t')
        d[en_id] = embedding

    return d


def read_ugd_json(file_name):
    data = pd.read_json(file_name)
    return data

def read_json(file_name):
    data = pd.read_json(file_name, encoding='utf-8')
    QA_paths = []
    
    for key, value in data.items():
        path_en_array = value['path_list']
        relations = value['relation_list']
        
        for path_ens in path_en_array:
            if len(path_ens) == 0:
                continue
            
            en_idx = 0
            # re_idx = 0
            path = []
            
            for en_idx in range(len(path_ens)):
                if en_idx == len(path_ens) - 1:
                    break
            
                head = path_ens[en_idx].encode('utf-8')
                tail = path_ens[en_idx + 1].encode('utf-8')
                
                triple = (head, relations, tail)
                path.append(triple)
                
            QA_paths.append(path)

    print("===reading QA finished===")
    return data, QA_paths

def read_emb(file_name):
    d = {}
    file = open(file_name, encoding='utf-8')
    for line in file:
        line = line.replace("[", "").replace("]", "")
        name, emb = line.strip().split('\t') 
        values_str = emb.split(',')   
        
        values_float = []
        for val in values_str:
            val = val.replace("np.float64(", "").replace(")", "")
            values_float.append(float(val))
        name = name.encode('utf-8')
         
        d[name] = torch.FloatTensor(values_float)
    file.close()
    
    return d


def load_data(model_args):
    global args, entity_dict, relation_dict, QA_json
    args = model_args 

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities("./data/MetaQA/KG_half/entities.dict")
    relation_dict = read_relations('./data/MetaQA/KG_half/relations.dict')
    
    ugd_drict = "./data/MetaQA/UGD/MetaQA" + '_' + args.hops + 'hop.json'

    QA_json, QA_paths = read_json(ugd_drict)

    print('reading train, validation, and test data ...')
    train_triplets = read_triplets("./data/" + args.QAdataset + '/' + args.KGdataset +'/train_idx.txt',
                                   "./data/" + args.QAdataset + '/' + args.KGdataset +'/train.txt', entity_dict)

    print('processing the knowledge graph ...')
    build_kg(train_triplets)


    return entity_dict, relation_dict, entity2edge_set, QA_json, QA_paths