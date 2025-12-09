import torch.nn.utils
from transformers import *
from vae.qa_vae import *

import torch
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor

class RelationExtractor(nn.Module):

    def __init__(self, entity_dim, latent_dim, relation_dim, ugd_json, num_entities, en_pretrained_embeddings,
                 re_pretrained_embeddings, en_emb_dict, re_emb_dict, re2idx, freeze, device, k_hop, entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0,
                 model='ComplEx', ls=0.0, do_batch_norm=True):

        super(RelationExtractor, self).__init__()

        self.entity_dim = entity_dim
        self.k_hop = k_hop
        self.current_path = []
        self.e_emb = torch.zeros(entity_dim)
        self.ugd_json = ugd_json
        self.re2idx = re2idx

        self.device = device
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        self.hidden_dim = 768
        self.relation_num = len(re_pretrained_embeddings)

        self.num_entities = num_entities

        self.bn2 = torch.nn.BatchNorm1d(entity_dim)

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        self.en_emb_dict = en_emb_dict
        self.re_emb_dict = re_emb_dict

        self.en_em_matrix = nn.Embedding.from_pretrained(torch.stack(en_pretrained_embeddings, dim=0))
        self.re_em_matrix = nn.Embedding.from_pretrained(torch.stack(re_pretrained_embeddings, dim=0))

        # self.en_em_matrix = nn.Embedding.from_pretrained(torch.FloatTensor(en_pretrained_embeddings), freeze=self.freeze)
        # self.re_em_matrix = nn.Embedding.from_pretrained(torch.FloatTensor(re_pretrained_embeddings),
        #                                                        freeze=self.freeze)
        self.lin1 = nn.Linear(self.hidden_dim, int(self.hidden_dim * 1.2))
        self.lin2 = nn.Linear(int(self.hidden_dim * 1.2), int(self.hidden_dim * 0.8))
        self.hidden2rel = nn.Linear(int(self.hidden_dim), int(relation_dim * k_hop))

        self.batchnorm1 = nn.BatchNorm1d(int(self.hidden_dim * 1.2))
        self.batchnorm2 = nn.BatchNorm1d(int(self.hidden_dim * 0.8))

        self.qa_vae_model = Qa_VAE(entity_dim, relation_dim, latent_dim, self.en_em_matrix,
                                   self.re_em_matrix, k_hop)

    def applyNonLinear(self, outputs):
        # outputs = self.lin1(outputs)
        # outputs = self.batchnorm1(outputs)
        # outputs = self.fcnn_dropout(outputs)
        # outputs = F.relu(outputs)
        #
        # outputs = self.lin2(outputs)
        # outputs = self.batchnorm2(outputs)
        # outputs = self.fcnn_dropout(outputs)
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        return outputs

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        return question_embedding

    def ComplEx_pre_rels(self, head, tail):
        head = self.en_emb_dict.get(head)
        tail = self.en_emb_dict.get(tail)

        head = self.ent_dropout(head.to(self.device))
        tail = self.ent_dropout(tail.to(self.device))

        re_head, im_head = torch.chunk(head, 2, dim=-1)  
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)  

        scores = []
        weights = self.re_em_matrix.weight
        for i in range(weights.size(0)):
            re_emb = weights[i]
            re_relation, im_relation = torch.chunk(re_emb, 2, dim=-1)

            re_score = re_head * re_relation - im_head * im_relation  
            im_score = re_head * im_relation + im_head * re_relation  

            final_score = torch.mm(re_score.unsqueeze(0), re_tail.unsqueeze(1)) + torch.mm(im_score.unsqueeze(0),
                                                                                           im_tail.unsqueeze(1))
            scores.append(final_score.item())

        scores = torch.tensor(scores)

        predicted_relation_id = torch.argmax(scores).item()

        return predicted_relation_id  # This will return the probability as a float value

    def ComplEx(self, head, relation, tail):
        head = self.en_emb_dict.get(head)
        tail = self.en_emb_dict.get(tail)

        head = self.ent_dropout(head.to(self.device))
        relation = self.rel_dropout(relation.to(self.device))
        tail = self.ent_dropout(tail.to(self.device))

        # Split the embeddings into real and imaginary parts (each tensor is split into two halves)
        re_head, im_head = torch.chunk(head, 2, dim=-1)  # Split head into real and imaginary parts
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)  # Split relation into real and imaginary parts
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)  # Split tail into real and imaginary parts

        # Compute real and imaginary parts of the score using the ComplEx scoring formula
        re_score = re_head * re_relation - im_head * im_relation  # Real part score
        im_score = re_head * im_relation + im_head * re_relation  # Imaginary part score

        # Stack the real and imaginary parts of the score
        score = torch.stack([re_score, im_score], dim=0)

        # Apply dropout to the score (this step is optional, can be removed if not needed)
        score = self.score_dropout(score)

        # Compute the final score by projecting into the tail embedding space
        final_score = torch.mm(re_score.unsqueeze(0), re_tail.unsqueeze(1)) + torch.mm(im_score.unsqueeze(0),
                                                                                       im_tail.unsqueeze(1))
        # Apply sigmoid to get the probability between 0 and 1
        prob = torch.sigmoid(final_score)

        # Return the probability as a scalar value (prob is a tensor with one element)
        prob_value = prob.item()

        return prob_value  # This will return the probability as a float value


    def forward(self, question_tokenized, attention_mask, path_emb_list,
                path_en_embs, neighbor_rels_onehot, epoch, total_epoch):
        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        rel_embedding = self.applyNonLinear(question_embedding)

        vae_loss = 0

        rel_num = len(path_emb_list)

        en_embs = torch.zeros(self.entity_dim)

        # gererate rel iteretely
        for i in range(rel_num):
            en_embs = path_en_embs[:, i, :]  # [batch_size, en_dim]
            part_en2 = path_en_embs[:, i + 1, :]  # [batch_size, en_dim]

            en_embs = torch.cat((en_embs, part_en2), dim=-1)

            if i > 0:
                en_embs = torch.cat((en_embs, path_emb_list[i - 1]), dim=-1)

            paths_rec, mu, log_var = self.qa_vae_model(en_embs, neighbor_rels_onehot)


            vae_loss += self.qa_vae_model.loss_function(paths_rec, path_emb_list[i],
                                    mu, log_var, epoch, total_epoch)

        qa_path_emb = torch.cat(path_emb_list, dim = 1)
        qa_cosine_loss = 1 - F.cosine_similarity(rel_embedding, qa_path_emb, dim=-1).mean()
        total_loss = vae_loss + qa_cosine_loss
        return total_loss

    def generate_k_hop_paths_parallel(self, query, k, cands, node_rels_map, threshold=0.1, mode=False):
        """
        Generate a k-hop path from the query to the set of candidate entities,
        where each hop consists of an entity and a relation.
        """

        vector_list = []
        if query in self.en_emb_dict:
            query_emb = self.en_emb_dict[query]
            vector_list.append(query_emb.to(self.device))
        else:
            vector_list.append(query)

        self.e_emb = torch.cat(vector_list, dim=0)
        all_paths_final = []

        def path_expansion_with_permutation(perm):
            current_path = [query]

            for first_entity in perm:
                for i in range(len(current_path) - 1, -1, -1):
                    if isinstance(current_path[i], str):
                        last_entity = current_path[i]
                        break

                current_path.append(first_entity)
                ne_emb = self.en_emb_dict.get(first_entity)
                if ne_emb is None:
                    continue

                tmp_emb = torch.cat((self.e_emb.to(self.device), ne_emb.to(self.device)), dim=0)
                tmp_emb = tmp_emb.unsqueeze(0)

                with torch.no_grad():
                    paths_rec, mu, log_var = self.qa_vae_model(tmp_emb.to(self.device), node_rels_map)

                if mode:
                    paths_rec = paths_rec.squeeze()

                    if last_entity not in self.en_emb_dict:
                        print()

                    score = self.ComplEx(last_entity, paths_rec, first_entity)
                    if score > threshold:
                        current_path.append(paths_rec)
                    else:
                        return
                else:
                    if paths_rec is not None:
                        current_path.append(paths_rec.squeeze())

            all_paths_final.append(current_path)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for perm in permutations(cands, k):
                futures.append(executor.submit(path_expansion_with_permutation, perm))

            for future in futures:
                future.result()

        return all_paths_final

    def sigmoid_distance(self, distance, scale=1.0):
        return 1 / (1 + torch.exp(-distance / scale))

    def euclidean_distance(self, qa_vector, vae_vector):
        qa_vector = qa_vector / qa_vector.norm(p=2, dim=-1, keepdim=True)
        vae_vector = vae_vector / vae_vector.norm(p=2, dim=-1, keepdim=True)
        distance = torch.norm(qa_vector - vae_vector)
        return self.sigmoid_distance(distance)


    def get_score_ranked(self, topic, question_tokenized, attention_mask, cans, k_hop, node_rels_map, KG_model=False):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0),
                                                       attention_mask.unsqueeze(0))
        threshold = 0.5
        # all_paths = []
        self.current_path = [topic]
        node_rels_map = node_rels_map.unsqueeze(0)

        all_paths = self.generate_k_hop_paths_parallel(query=topic, k=k_hop, cands=cans,
                                                       node_rels_map=node_rels_map)
        rel_embedding = self.applyNonLinear(question_embedding)

        can_anws = []
        for path in all_paths:
            if len(path) != k_hop * 2 + 1:
                continue

            rels = []
            p_ens = []
            for item in path:
                if isinstance(item, str):
                    p_ens.append(item)
                else:
                    rels.append(item)

            rel_embedding = rel_embedding.squeeze(0)
            rel_tensor = torch.cat(rels, dim=-1)

            similarity = F.cosine_similarity(rel_embedding, rel_tensor, dim=-1).item()

            if similarity > threshold:
                if p_ens[-1] not in can_anws:
                    can_anws.append(p_ens[-1])

        return can_anws


    def get_score_ranked_complEx(self, topic, question_tokenized, attention_mask, cans, k_hop, node_rels_map, KG_model=False):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0),
                                                       attention_mask.unsqueeze(0))
        threshold = 0.5
        self.current_path = [topic]
        node_rels_map = node_rels_map.unsqueeze(0)

        all_paths = self.generate_k_hop_paths_parallel_complEx(query=topic, k=k_hop, cands=cans,
                                                       node_rels_map=node_rels_map)

        rel_embedding = self.applyNonLinear(question_embedding)

        can_anws = []
        for path in all_paths:
            if len(path) != k_hop * 2 + 1:
                continue

            rels = []
            p_ens = []
            for item in path:
                if isinstance(item, str):
                    p_ens.append(item)
                else:
                    rels.append(item)

            relations = self.ugd_json[topic]['relation_list']
            rel_id_trues = []
            if isinstance(relations, str):
                rel_true = self.ugd_json[topic]['relation_list']
                rel_id_trues = self.re2idx[rel_true]

                if rel_id_trues == rels[0]:
                    if p_ens[-1] not in can_anws:
                        can_anws.append(p_ens[-1])
            else:
                for rel in list(self.ugd_json[topic]['relation_list']):
                    rel_id_trues.append(self.re2idx[rel])

                if rel_id_trues == rels:
                    if p_ens[-1] not in can_anws:
                        can_anws.append(p_ens[-1])

        return can_anws
