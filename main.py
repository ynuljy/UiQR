import os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataloader import DatasetMetaQA
from model import RelationExtractor
import sys
import ast
from candidates.can_en import *
from multiprocessing import freeze_support

sys.path.append("../../")  # Adds higher directory to python modules path.


def read_ugd_json(file_name):
    ugd_json = pd.read_json(file_name)
    return ugd_json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return True


parser = argparse.ArgumentParser()

# ======base_args====== #
parser.add_argument('--QAdataset', type=str, default='MetaQA', help='QA dataset name')
parser.add_argument('--KGdataset', type=str, default='KG_full', help='KG dataset name')
# parser.add_argument('--hops', type=str, default='1')

# ======vae_args====== #
parser.add_argument('--entity_dim', type=int, default=400, help='entity dim')
parser.add_argument("--relation_dim", type=int, default=400)
parser.add_argument('--latent_dim', type=int, default=200, help='latent dim')

# ======qa_args====== #
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--shuffle_data", type=bool, default=True)
parser.add_argument("--num_workers", type=int, default=0)
# parser.add_argument("--neg_batch_size", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=200)
parser.add_argument("--use_cuda", type=bool, default=True)

# --do_batch_norm 1
parser.add_argument("--do_batch_norm", type=int, default=256)
# --gpu 0
parser.add_argument("--gpu", type=int, default=0)
# --freeze 1
parser.add_argument("--freeze", type=int, default=True)
# --batch_size 16
parser.add_argument("--batch_size", type=int, default=256)
# --validate_every 5
parser.add_argument("--validate_every", type=int, default=2)
# --lr 0.00002
parser.add_argument("--lr", type=float, default=0.01)
# --entdrop 0.0
parser.add_argument("--entdrop", type=float, default=0.0)
# --reldrop 0.0
parser.add_argument("--reldrop", type=float, default=0.0)
# --scoredrop 0.0
parser.add_argument("--scoredrop", type=float, default=0.0)
# --decay 1.0
parser.add_argument("--decay", type=float, default=1.0)
# --model ComplEx
parser.add_argument("--model", type=str, default="ComplEx")
# --patience 20
parser.add_argument("--patience", type=int, default=20)
# --ls 0.05
parser.add_argument("--ls", type=float, default=0.05)
# --l3_reg 0.001
parser.add_argument("--l3_reg", type=float, default=0.001)

#####
MODE = 'train'

# --mode train
parser.add_argument("--mode", type=str, default=MODE)
# --hops "$hops"
parser.add_argument("--hops", type=int, default="3")
# --nb_epochs "$nb_epochs"
parser.add_argument("--nb_epochs", type=int, default=20)
# --outfile "$outfile"
parser.add_argument("--outfile", type=str, default="meta_qa_half")
parser.add_argument("--load_from", type=str, default="")

# --candidate size
parser.add_argument("--can_size", type=int, default="10")

# 解析传入的参数
args = parser.parse_args()


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def read_kg(file_name):
    node_rels_map = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ws = line.strip().split("\t")
            if len(ws) < 1:
                continue
            if ws[0] not in node_rels_map:
                node_rels_map[ws[0]] = set()
            if ws[2] not in node_rels_map:
                node_rels_map[ws[2]] = set()

            node_rels_map[ws[0]].add(ws[1])
            node_rels_map[ws[2]].add(ws[1])

    return node_rels_map


def relToOneHot2(indices, relation2idx):
    # sample 4 indices here
    num_rel = len(indices)
    indices = torch.LongTensor(indices)
    vec_len = len(relation2idx)
    one_hot = torch.FloatTensor(vec_len)
    one_hot.zero_()
    # one_hot = -torch.ones(vec_len, dtype=torch.float32)
    one_hot.scatter_(0, indices, 1)
    return one_hot


def validate_v2(model, data_path, entity2idx, ugd_json, relation2idx,
                dataloader, device, k_hop, canE, node_rels_map, writeCandidatesToFile=True, output_topk=True):
    model.eval()
    data = process_text_file(data_path, ugd_json)
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key

    idx2relation = {}
    for key, value in relation2idx.items():
        idx2relation[value] = key

    data_gen = data_generator(data=data, dataloader=dataloader,
                              entity2idx=entity2idx, node_rels_map=node_rels_map, relation2idx=relation2idx)

    total_correct = 0
    num_incorrect = 0
    hit_at_10 = 0

    for i in tqdm(range(len(data))):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        query = d[1]

        if query == 'Love Letter':
            print()

        question_tokenized = d[2].to(device)
        attention_mask = d[3].to(device)
        ans = d[4]

        rel_onehot = d[6].to(device)
        cans = canE.greedy_search(query)

        pred_ans = model.get_score_ranked(
            topic=query,
            question_tokenized=question_tokenized,
            attention_mask=attention_mask,
            cans=cans,
            k_hop=k_hop,
            node_rels_map=rel_onehot
        )

        # pred_ans = model.get_score_ranked_complEx(
        #     topic=query,
        #     question_tokenized=question_tokenized,
        #     attention_mask=attention_mask,
        #     cans=cans,
        #     k_hop=k_hop,
        #     node_rels_map=rel_onehot
        # )

        if type(ans) is int:
            ans = [ans]
        is_correct = False
        for pred_an in pred_ans:
            if pred_an in ans:
                is_correct = True
                total_correct += 1
                break
        if not is_correct:
            num_incorrect += 1

    print(hit_at_10 / len(data))
    accuracy = total_correct / len(data)

    return accuracy


def getEmbeddings_2(file_name):
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

        d[name] = torch.FloatTensor(values_float)

    file.close()
    return d


def getEntityEmbeddings(kge_model):
    e = {}
    entity_dict = './data/MetaQA/embeddings/ComplEx_MetaQA_full/entities.dict'

    f = open(entity_dict, 'r', encoding='utf-8')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        # e[ent_name] = kge_model[ent_id]
        e[ent_name] = torch.tensor(kge_model[ent_id], dtype=torch.float)
    f.close()
    return e


def getRelationEmbeddings(kge_model):
    r = {}
    relation_dict = './data/MetaQA/KG_full/relations.dict'

    f = open(relation_dict, 'r', encoding='utf-8')
    for line in f:
        line = line[:-1].split('\t')
        rel_id = int(line[1])
        rel_name = line[0]
        r[rel_name] = torch.tensor(kge_model[rel_id], dtype=torch.float)
    f.close()
    return r


def process_text_file(text_file, ugd_json, split=False):
    data_file = open(text_file, "r", encoding='utf-8')
    data_array = []

    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == "":
            continue
        data_line = data_line.strip().split("\t")
        # if no answer
        if len(data_line) != 2:
            continue

        question = data_line[0].split("[")
        question_1 = question[0]
        question_2 = question[1].split("]")
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1 + "NE" + question_2
        ans = data_line[1].split("|")

        if head not in ugd_json:
            continue

        flag = False

        ugd_ans = ugd_json[head]['answer_list'][0]

        answer_list = ast.literal_eval(ugd_ans)
        for an in ans:
            if an in answer_list:
                flag = True
                break
        if not flag:
            continue

        entity_count_dict = ugd_json[head]['entity_count_dict']
        if len(entity_count_dict) > 15:
            continue

        try:
            # get rel_groudtruth
            relations = ugd_json[head]['relation_list']
            if isinstance(relations, str):
                rels = ugd_json[head]['relation_list']
            else:
                rels = list(ugd_json[head]['relation_list'])

            # get path_ens_groudtruth
            path_ens_list = ugd_json[head]['path_list']

            for path_ens in path_ens_list:
                data_array.append([head, question.strip(), ans, rels, path_ens, data_line[-1]])
        except KeyError:
            # print(head)
            continue

    if split == False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data


def data_generator(data, dataloader, entity2idx, node_rels_map, relation2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = data_sample[0].strip()
        head_inx = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            # TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity.strip())

        neighbors = node_rels_map[data_sample[0].strip()]
        neighbor_ids = []
        for n in neighbors:
            if n in relation2idx:
                neighbor_ids.append(relation2idx[n])
        neighbor_rel_onehot = relToOneHot2(neighbor_ids, relation2idx)

        yield torch.tensor(
            head_inx, dtype=torch.long
        ), head, question_tokenized, attention_mask, ans, data_sample[1], neighbor_rel_onehot


def train(data_path,
        entity_dim,
        latent_dim,
        batch_size,
        nb_epochs,
        relation_dim,
        gpu,
        use_cuda,
        freeze,
        entdrop,
        reldrop,
        scoredrop,
        l3_reg,
        model_name,
        decay,
        ls,
        load_from,
        do_batch_norm,
        validate_every,
        patience,
        k_hop,
        can_size
):
    print("Loading entities and relations")
    current_path = os.getcwd()
    directory = current_path + "/data/" + args.QAdataset

    kge_model_en = np.load('./data/MetaQA/embeddings/ComplEx_MetaQA_full/E.npy')  # entity embedding
    kge_model_re = np.load('./data/MetaQA/embeddings/ComplEx_MetaQA_full/R.npy')  # relation embdedding

    en_emb_dict = getEntityEmbeddings(kge_model_en)
    re_emb_dict = getRelationEmbeddings(kge_model_re)

    ugd_json = read_ugd_json(directory + '/UGD/MetaQA_' + str(k_hop) + 'hop.json')

    print("Loaded entities and relations")

    entity2idx, idx2entity, en_embed_matrix = prepare_embeddings(en_emb_dict)
    re2idx, idx2re, re_embed_matrix = prepare_embeddings(re_emb_dict)

    data = process_text_file(data_path, ugd_json, split=False)
    print("Train file processed, making dataloader")

    node_rels_map = read_kg(directory + '/' + args.KGdataset + '/train.txt')

    en_emb_dict_2 = getEmbeddings_2(directory + '/cache/KG_full/entity_128dim_batch400.txt')
    entity2idx_2, idx2entity_2, en_embed_matrix_2 = prepare_embeddings(en_emb_dict_2)
    canEn = CanEn(en_emb_dict_2, idx2entity_2, entity2idx_2, ugd_json)

    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, en_emb_dict, re_emb_dict, entity2idx, re2idx, entity_dim, node_rels_map)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    print("Creating model...")
    model = RelationExtractor(
        entity_dim,
        latent_dim,
        relation_dim,
        ugd_json=ugd_json,
        num_entities=len(idx2entity),
        en_pretrained_embeddings=en_embed_matrix,
        re_pretrained_embeddings=re_embed_matrix,
        en_emb_dict=en_emb_dict,
        re_emb_dict=re_emb_dict,
        re2idx=re2idx,
        freeze=freeze,
        device=device,
        k_hop=k_hop,
        entdrop=entdrop,
        reldrop=reldrop,
        scoredrop=scoredrop,
        l3_reg=l3_reg,
        model=model_name,
        ls=ls,
        do_batch_norm=do_batch_norm
    )

    print("Model created!")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0

    for epoch in range(nb_epochs):
        phases = ['train'] * validate_every + ['valid']

        for phase in phases:
            if phase == 'train':
                model.train()

                loader = tqdm(data_loader, total=len(data_loader), unit="batches",
                              desc="Epoch {}/{}".format(epoch, nb_epochs))

                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2]
                    # positive_rels = a[3].to(device)
                    positive_rels = [t.to(device) for t in a[3]]
                    path_en_embs = a[4].to(device)
                    neighbor_rels_onehot = a[5].to(device)

                    loss = model(question_tokenized=question_tokenized,
                                 attention_mask=attention_mask,
                                 path_emb_list=positive_rels,
                                 path_en_embs=path_en_embs,
                                 neighbor_rels_onehot=neighbor_rels_onehot,
                                 epoch=epoch,
                                 total_epoch=nb_epochs)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                    loader.set_postfix(loss=loss.item())

            elif phase == 'valid':
                print('=======valid======')
                model.eval()
                eps = 0.001

                score = validate_v2(model=model, data_path=valid_data_path, entity2idx=entity2idx, ugd_json=ugd_json,
                                    relation2idx=re2idx, dataloader=dataset, device=device, k_hop=k_hop, canE=canEn,
                                    node_rels_map=node_rels_map)

                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update += 1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check" % (
                    score, best_score, patience - no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    exit()
                if epoch == nb_epochs - 1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    exit()

                print(f'Epoch [{epoch}/{nb_epochs}], '
                      f'Accuracy: {score:.4f}, '
                      f'Mode: {phase}')


if __name__ == '__main__':
    freeze_support()

    data_path = './data/MetaQA/QA/qa_train_1hop.txt'
    valid_data_path = './data/MetaQA/QA/qa_dev_1hop.txt'
    test_data_path = './data/MetaQA/QA/qa_test_1hop.txt'

    if args.mode == "train":
        train(data_path, args.entity_dim, args.latent_dim, args.batch_size,
              args.nb_epochs, args.relation_dim, args.gpu, args.use_cuda,
              args.freeze, args.entdrop, args.reldrop, args.scoredrop, args.l3_reg,
              args.model, args.decay, args.ls, args.load_from, args.do_batch_norm,
              args.validate_every, args.patience, args.hops, args.can_size)

    elif args.mode == "test":
        eval(
            data_path=test_data_path,
            load_from=args.load_from,
            gpu=args.gpu,
            hidden_dim=args.hidden_dim,
            relation_dim=args.relation_dim,
            embedding_dim=args.embedding_dim,
            hops=args.hops,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_name=args.model,
            do_batch_norm=args.do_batch_norm,
            use_cuda=args.use_cuda,
        )
