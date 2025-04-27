import numpy as np
import copy
import torch
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import math


class CanEn:
    def __init__(self, en_emb_dict, idx2entity, entity2idx, ugd_json):
        self.ugd_json = ugd_json
        self.en_emb_dict = en_emb_dict
        self.idx2entity = idx2entity
        self.entity2idx = entity2idx
        self.cos_matrix = self.get_cos_matrix()

    def getCosineSim(self, en_a_idx, en_b_idx):
        en_a = self.idx2entity[en_a_idx]
        en_b = self.idx2entity[en_b_idx]

        vector_en_a = self.en_emb_dict[en_a]
        vector_en_b = self.en_emb_dict[en_b]

        simi_i = np.dot(vector_en_a, vector_en_b) / (np.linalg.norm(vector_en_a) * (np.linalg.norm(vector_en_b)))

        return simi_i

    def mutual_information(self, en_a, en_b, entity_count_dict):
        if (en_a not in entity_count_dict) or (en_b not in entity_count_dict):
            return 0

        total_count = sum(entity_count_dict.values())

        entity_vector = {entity: count / total_count for entity, count in entity_count_dict.items()}

        freq1 = entity_vector.get(en_a, 0.0)
        freq2 = entity_vector.get(en_b, 0.0)

        similarity = freq1 * freq2

        if similarity > 0:
            score = -math.log(1 - similarity)
        else:
            score = 0.0

        return score


    def weighted_from_d(self, en_a, en_b, entity_count_dict):
        '''
        the function calculate the weight between the query and other entities in UGD
        :return:
        '''
        return self.mutual_information(en_a, en_b, entity_count_dict)

    def weighted_from_kg(self, en_a_idx, en_b_idx):
        '''
        the function calculate the weight between the query and other entities in KG
        :return:
        '''
        return self.getCosineSim(en_a_idx, en_b_idx)

    def normal_d_kg(self, en_a, en_b, entity_count_dict):
        '''
        the function normalizes the entity weight of kg and data
        :return:
        '''
        d_score = self.weighted_from_d(en_a, en_b, entity_count_dict)

        en_a_idx = self.entity2idx[en_a]
        en_b_idx = self.entity2idx[en_b]
        kg_score = self.cos_matrix[en_a_idx, en_b_idx]
        score = kg_score + d_score * 10

        return score

    def get_cos_matrix(self):
        embeddings = np.array(list(self.en_emb_dict.values()))
        score_matrix = cosine_similarity(embeddings, embeddings)

        return score_matrix

    def submodular_function2(self, S, entity_count_dict):
        total_score = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if S_list[i] != S_list[j]:
                    total_score += self.normal_d_kg(S_list[i], S_list[j], entity_count_dict)
        return total_score

    def submodular_function(self, query, S, entity_count_dict):
        total_score = 0
        S_list = list(S)
        for i in range(len(S_list)):
            if S_list[i] == query:
                continue
            total_score += self.normal_d_kg(query, S_list[i], entity_count_dict)

        return total_score

    def greedy_search(self, query):
        Ec = set()
        Ec.add(query)

        entity_count_dict = self.ugd_json[query]['entity_count_dict']
        remaining_indices = list(copy.deepcopy(entity_count_dict))

        if query in entity_count_dict:
            remaining_indices.remove(query)

        k=10

        def marginal_gain(element):
            new_set = Ec | {element}
            return self.submodular_function(query, new_set, entity_count_dict) \
                   - self.submodular_function(query, Ec, entity_count_dict)

        for _ in range(k):
            if len(remaining_indices) == 0:
                break
            best_element = max(remaining_indices, key=marginal_gain)
            Ec.add(best_element)
            remaining_indices.remove(best_element)

        Ec.remove(query)
        return Ec


    def submudle_fun(self, subm_v, can_en_list):
        subm_score_dict = {}
        score = 0
        if subm_v not in can_en_list:
            for can in can_en_list:
                subm_score_dict[can] = self.cache[self.map_index[subm_v]][self.map_index[can]]
            tmp_list_max_result = max(subm_score_dict.keys(), key=lambda x: subm_score_dict[x])
            score += subm_score_dict[tmp_list_max_result]
        return score

    def records_sorted(self, query_en, file):
        records_dict = {}
        for record in self.record_entity_labels:
            if int(record) == query_en:
                continue
            records_dict[record] = self.normal_d_kg(int(query_en), record)

        sorted_result = sorted(records_dict.items(), key=lambda x: x[1], reverse=True)
        for re, score in sorted_result:
            file.write(str(re) + "\t" + str(score) + "\n")
        file.close()
