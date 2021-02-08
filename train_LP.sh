lr=0.01
steps=10
eps=0.03
dataset=cifar10
network=wide
data_root=./datasets
epoch=60
attack=pgd
save_dir=./experiment

CUDA_VISIBLE_DEVICES=2 python3 ./train/train_LP.py \
                            --lr $lr \
                            --steps $steps \
                            --eps $eps \
                            --dataset $dataset \
                            --network $network \
                            --data_root $data_root \
                            --epoch $epoch \
                            --attack $attack \
                            --save_dir $save_dir
