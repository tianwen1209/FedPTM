# FedPTM
fedrated learning for pre-trained models

## related work
[FedPCL](https://arxiv.org/pdf/2209.10083.pdf)

## Commands to run FedPCL
```bash
python3.9 exps/federated_main.py --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedavg_fnli_1bb_5u.log
```
configs:
```bash
--feature_iid 0
--label_iid 1
```

## Fine-pruning Defense
code for [Fine-pruning](https://github.com/kangliucn/Fine-pruning-defense)
paper for [Fine-pruning](https://arxiv.org/pdf/2011.01767.pdf)

## Draw activation map
[code](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keract-activation.ipynb)
 
