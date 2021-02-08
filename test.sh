#!/bin/bash

# 0: plain, 1: AT, 2: RFM, 3: mAT, 4: LP, 5: trade
number=0
cuda_id=0
dataset=cifar10
network=wide

steps=10
eps=0.08
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

   datetime=02010234
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
