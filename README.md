# Orthogonal Embedded Memory (OEM)
*Author  : `Byung-Kwan Lee` (leebk@kaist.ac.kr),    Ph.D. Candidate, Electrical Engineering, KAIST*

## Adversarial Defense Contents in train folder
* **OEM**    (AdversarialMemory)
    - Run train_OEM.sh
* **TRADE**     (Adversarial KLDiv training with clean examples)
    - Run train_trade.sh
* **LP**     (Adversarial Logit Pairing with clean examples)
    - Run train_LP.sh
* **mAT**     (Adversarial Training with clean examples)
    - Run train_mAT.sh
* **AT**     (Adversarial Training)
    - Run train_AT.sh
* **Plain**  (Plain Training w/o Adversarial examples)
    - Run train_plain.sh

### Run Training
For example: train_OEM.sh
```bash
sh train_OEM.sh
```
```bash
lr=0.01
steps=10
eps=0.03
dataset=cifar10
network=wide
data_root=./datasets
epoch=1
attack=pgd
save_dir=./experiment
prior=1
prior_datetime=01051206

CUDA_VISIBLE_DEVICES=0 python3 ./train/train_OEM.py \
                            --lr $lr \
                            --steps $steps \
                            --eps $eps \
                            --dataset $dataset \
                            --network $network \
                            --data_root $data_root \
                            --epoch $epoch \
                            --attack $attack \
                            --save_dir $save_dir \
                            --prior $prior \
                            --prior_datetime $prior_datetime
```

### Testing
```bash
sh test.sh
```
```bash
#!/bin/bash

# 0: plain, 1: AT, 2: OEM, 3: mAT, 4: LP, 5: trade
number=0
cuda_id=0
dataset=cifar10
network=wide

steps=10
eps=0.03
data_root=./datasets
save_dir=./experiment


if [ ${number} -eq 5 ] ;
then

   datetime=01051206
   baseline=TRADE

elif [ ${number} -eq 4 ] ;
then

   datetime=01051206
   baseline=LP

elif [ ${number} -eq 3 ] ;
then

   datetime=01051206
   baseline=mAT

elif [ ${number} -eq 2 ] ;
then

   datetime=01051206
   baseline=OEM

elif [ ${number} -eq 1 ] ;
then

   datetime=01051206
   baseline=AT
else
  datetime=01051206
  baseline=Plain
fi

CUDA_VISIBLE_DEVICES=${cuda_id}   python3 ./test/test.py \
                            --steps $steps \
                            --eps $eps \
                            --dataset $dataset \
                            --network $network \
                            --data_root $data_root \
                            --save_dir $save_dir \
                            --datetime $datetime \
                            --baseline $baseline

```