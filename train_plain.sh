lr=0.01
dataset=cifar10
network=vgg
data_root=./datasets
epoch=60
save_dir=./experiment

CUDA_VISIBLE_DEVICES=2 python3 ./train/train_plain.py \
                            --lr $lr \
                            --dataset $dataset \
                            --network $network \
                            --data_root $data_root \
                            --epoch $epoch \
                            --save_dir $save_dir
