#!/bin/bash

echo "Executing the training scripts"

# this script will computed adjusted cosine similarity and L2 distance of feature vectors

#load resnet18 mnist, compute adj_cos sim and L2 on mnist
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset mnist  --adj_cos_sim --l2 --desc "load mnist resnet18, compute adj_cos_sim and L2 on mnist"

#load resnet18 mnist, compute adj_cos sim and L2 on fashion_mnist
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset mnist  --adj_cos_sim --l2 --cs_dataset fashion_mnist --desc "load mnist resnet18, compute adj_cos_sim and L2 on fashion_mnist"

#load resnet18 fashion_mnist, compute adj_cos sim and L2 on fashion_mnist
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset fashion_mnist  --adj_cos_sim --l2 --desc "load fashion_mnist resnet18, compute adj_cos_sim and L2 on fashion_mnist"

#load resnet18 fashion_mnist, compute adj_cos sim and L2 on mnist
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset fashion_mnist  --adj_cos_sim --l2 --cs_dataset mnist --desc "load fashion_mnist resnet18, compute adj_cos_sim and L2 on mnist"

#load resnet18 cifar10, compute adj_cos sim and L2 on cifar10
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset cifar10  --adj_cos_sim --l2 --desc "load cifar10 resnet18, compute adj_cos_sim and L2 on cifar10"

#load resnet18 cifar10, compute adj_cos sim and L2 on cifar100
python Scripts/script.py --execution_id "" --cuda --model resnet18 --train_dataset cifar10  --adj_cos_sim --l2 --cs_dataset cifar100 --desc "load cifar10 resnet18, compute adj_cos_sim and L2 on cifar100"

echo "All the scripts have been executed"