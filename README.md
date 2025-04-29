# Instructions

## Data and pre-trained models
In order to run the code, first download data.zip and pretrained_model.zip from [here]([https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing](https://drive.google.com/drive/folders/1Iqj9I3RMr-8vQtqSXsDp6-GNcT3UiHP-?dmr=1&ec=wgc-drive-globalnav-goto)). Unzip these files in the main directory.

Change to directory ./UiQR/data/. 

Following is an example command to embed KG by running the training code
```
python .\kge\main.py
```

Following is an example command to run the training code of UiQR
```
python main.py
```
### QA Dataset

There are 5 files for each dataset (1, 2 and 3 hop)
- qa_train_{n}hop_train.txt
- qa_train_{n}hop_train_half.txt
- qa_train_{n}hop_train_old.txt
- qa_dev_{n}hop.txt
- qa_test_{n}hop.txt

Out of these, qa_dev, qa_test and qa_train_{n}hop_old are exactly the same as the MetaQA original dev, test and train files respectively.

For qa_train_{n}hop_train and qa_train_{n}hop_train_half, we have added triple (h, r, t) in the form of (head entity, question, answer). This is to prevent the model from 'forgetting' the entity embeddings when it is training the QA model using the QA dataset. qa_train.txt contains all triples, while qa_train_half.txt contains only triples from MetaQA_half.

## WebQuestionsSP

### KG dataset

There are 2 datasets: fbwq_full and fbwq_half

Creating fbwq_full: We restrict the KB to be a subset of Freebase which contains all facts that are within 2-hops of any entity mentioned in the questions of WebQuestionsSP. We further prune it to contain only those relations that are mentioned in the dataset. This smaller KB has 1.8 million entities and 5.7 million triples. 

Creating fbwq_half: We randomly sample 50% of the edges from fbwq_full.

### QA Dataset

Same as the original WebQuestionsSP QA dataset.

## SimpleQA

### KG dataset

Same as the original WebQuestionsSP KG dataset.

Creating fbwq_full: We restrict the KB to be a subset of Freebase which contains all facts that are within 2-hops of any entity mentioned in the questions of WebQuestionsSP. We further prune it to contain only those relations that are mentioned in the dataset. This smaller KB has 1.8 million entities and 5.7 million triples. 

Creating fbwq_half: We randomly sample 50% of the edges from fbwq_full.

### QA Dataset

Same as the original SimpleQA QA dataset.
