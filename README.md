# FedPTM
fedrated learning for pre-trained models

## related work
[FedPCL](https://arxiv.org/pdf/2209.10083.pdf)

## Commands to prepare own PTM
[code for pre-train models](https://github.com/mboudiaf/pytorch-meta-dataset/tree/master#13-download-pre-trained-models)

### prepare dataset
To download the META-DATASET, please follow the details instructions provided at [meta-dataset](https://github.com/google-research/meta-dataset/blob/main/doc/dataset_conversion.md#notes) to obtain the .tfrecords converted data. Once done, make sure all converted dataset are in a single folder, and execute the following to produce index files:
```bash
make index_files
```
To convert dataset:
```bash
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$DATASRC//home/yinqiu/meta-dataset/aircraft/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS
```
#### Bugs in meta-dataset:
![6561685868068_ pic](https://github.com/tianwen1209/FedPTM/assets/78245339/852f6baa-acd1-4a50-b645-b2095928cdff)
``` python
bbox = list(bboxes[i])if bboxesis not None else None
```


Then exports:
```bash
export RECORDS='path/to/records'
```

### Train models from scratch (optional)
```bash
make method=<method> arch=<architecture> base=<base_dataset> val=<validation_dataset> train
# make method=simpleshot arch=resnet18 base=quickdraw val=quickdraw train
```

## Commands to run FedPCL
```bash
conda activate PTM_tiantian
python exps/federated_main.py --alg fedpcl --dataset office --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >digit_fedpcl_fnli_1bb_5u.log
```

configs:
```bash
--feature_iid 1
--label_iid 0
```

## black-doored methods and datasets
path of backdoor with 20 classes:
```bash
 /home/yinqiu/pytorch-meta-dataset/checkpoints/base=cu_birds/val=cu_birds/arch=resnet18/method=standard/checkpoint.pth.tar
```

## Fine-pruning Defense
code for [Fine-pruning](https://github.com/kangliucn/Fine-pruning-defense)

paper for [Fine-pruning](https://arxiv.org/pdf/2011.01767.pdf)

## Draw activation map
[code](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keract-activation.ipynb)
 
## Predict Euclidean distance
[code & paper](https://github.com/mdshihabullah/federated-predicted-euclidean-distance/blob/main/thesis_research_report.pdf)

