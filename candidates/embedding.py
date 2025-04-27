import codecs
import random
import math
import numpy as np
import copy
import time

def distanceL2(h,t):
    return np.sum(np.square(h - t))

def distanceL1(h,t):
    return np.sum(np.fabs(h - t))

class Embedding:
    def __init__(self, entity2edge_set, entity_set, relation_set, en_pair_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.en_pair_list = en_pair_list
        self.L1=L1

        self.entity2edge_set = entity2edge_set
        
        self.loss = 0

    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) ,
                                           6/math.sqrt(self.embedding_dim) ,
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp,ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) ,
                                        6/math.sqrt(self.embedding_dim) ,
                                        self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp,ord=2)

        self.relation = relation_dict
        self.entity = entity_dict


    def aggregate_neighbors(self, entity_embeddings, neighbors, relation_dict):
        neighbors_embeddings = np.zeros_like(entity_embeddings) 
        if len(neighbors) == 0:
            return neighbors_embeddings   
         
        for neighbor in neighbors:  
            neighbors_embeddings += relation_dict[neighbor]  
            
        return neighbors_embeddings 

    def train(self, epochs):
        nbatches = 400
        batch_size = len(self.en_pair_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            for k in range(nbatches):
                # Sbatch:list
                Sbatch = random.sample(self.en_pair_list, batch_size)
                Tbatch = []

                for en_pair in Sbatch:
                    corrupted_triple = self.Corrupt(en_pair)
                    if (en_pair, corrupted_triple) not in Tbatch:
                        Tbatch.append((en_pair, corrupted_triple))
                self.update_embeddings(Tbatch)


            end = time.time()
            print("epoch: ", epoch , "cost time: %s"%(round((end - start),3)))
            print("loss: ", self.loss)

            if epoch % 20 == 0:
                with codecs.open("entity_temp.txt", "w", encoding='utf-8') as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp.txt", "w", encoding='utf-8') as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        print("writing embedding...")
        with codecs.open("../data/MetaQA/cache/KG_half/entity_128dim_batch400.txt", "w", encoding='utf-8') as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("../data/MetaQA/cache/KG_half/relation128dim_batch400.txt", "w", encoding='utf-8') as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("finish writing")


    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        if seed > 0.5:
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0]=rand_head

        else:
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            h_neighbors = self.aggregate_neighbors(h_correct, 
                                self.entity2edge_set[triple[0]], self.relation)            
            t_neighbors = self.aggregate_neighbors(t_correct, 
                                self.entity2edge_set[triple[1]], self.relation)

            h_err = self.entity[corrupted_triple[0]]
            h_err_neighbors = self.aggregate_neighbors(h_err, 
                                self.entity2edge_set[triple[0]], self.relation)

            t_err = self.entity[corrupted_triple[1]]
            t_err_neighbors = self.aggregate_neighbors(t_err, 
                                self.entity2edge_set[triple[1]], self.relation)

            if self.L1:
                dist_correct = distanceL1(h_correct + h_neighbors, t_correct + t_neighbors)
                dist_corrupt = distanceL1(h_corrupt + h_neighbors, t_corrupt + t_neighbors)
            else:
                dist_correct = distanceL2(h_correct + h_err_neighbors, t_correct + t_err_neighbors)
                dist_corrupt = distanceL2(h_corrupt + h_err_neighbors, t_corrupt + t_err_neighbors)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + h_neighbors - t_correct - t_neighbors)
                grad_neg = 2 * (h_corrupt + h_err_neighbors - t_corrupt - t_err_neighbors)

                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                if triple[0] == corrupted_triple[0]:
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[1] == corrupted_triple[1]:
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                for neighbor in self.entity2edge_set[triple[0]]:   
                    copy_relation[neighbor] -= self.learning_rate * grad_pos
                    copy_relation[neighbor] -= (-1)*self.learning_rate * grad_neg

        #batch norm
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct-dist_corrupt+self.margin)
