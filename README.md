# Instructions

## Data and pre-trained models
In order to run the code, first download data and pretrained_model from [here](https://drive.google.com/drive/folders/1Iqj9I3RMr-8vQtqSXsDp6-GNcT3UiHP-?dmr=1&ec=wgc-drive-globalnav-goto). Unzip these files in the main directory.

The resulting file structure will look like:

```plain
.
├── main
├── README.md
├── data/
    ├── metaQA/                 (prerocessed ConceptNet)
    ├── WQSP/
        ├── catch
        ├── KG_full
        ├── KG_half
        ├── pretrained_models             (converted statements)
        ├── QA/              (grounded entities)
        ├── UGD/                (extracted subgraphs)
    ├── SimpleQA/
├── candidates/
├── kge/
    ├── mian.py
└── vae/
```

Following is an example command to embed KG by running the training code
```
python .\kge\main.py
```

Following is an example command to run the training code of UiQR
```
python main.py
```
### UGD Dataset
UGD contains the entity co-occurrence frequency that we have processed and the KG multi-hop paths for query reasoning.


## SimpleQA

### KG dataset

Same as the original WQSP KG dataset.
