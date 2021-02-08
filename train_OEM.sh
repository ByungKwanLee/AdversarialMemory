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
