#!/bin/bash

echo "Executing the L2 scripts"

# this script will computed adjusted cosine similarity and L2 distance of feature vectors

#load resnet18 mnist, compute adj_cos sim and L2 on mnist
python Scripts/script.py --execution_id "2024-02-26-08-25-11" --cuda --model resnet18 --train_dataset mnist --cs_dataset mnist  --l2  --desc "load mnist resnet18, compute cos_sim_adj and L2 on mnist"

#load resnet18 mnist, compute adj_cos sim and L2 on fashion_mnist
python Scripts/script.py --execution_id "2024-02-26-08-25-11" --cuda --model resnet18 --train_dataset mnist  --l2  --cs_dataset fashion_mnist --desc "load mnist resnet18, compute cos_sim_adj and L2 on fashion_mnist"

#load resnet18 fashion_mnist, compute adj_cos sim and L2 on fashion_mnist
python Scripts/script.py --execution_id "2024-02-26-08-33-57" --cuda --model resnet18 --train_dataset fashion_mnist  --cs_dataset fashion_mnist --l2  --desc "load fashion_mnist resnet18, compute cos_sim_adj and L2 on fashion_mnist"

#load resnet18 fashion_mnist, compute adj_cos sim and L2 on mnist
python Scripts/script.py --execution_id "2024-02-26-08-33-57" --cuda --model resnet18 --train_dataset fashion_mnist  --l2  --cs_dataset mnist --desc "load fashion_mnist resnet18, compute cos_sim_adj and L2 on mnist"

#load resnet18 cifar10, compute adj_cos sim and L2 on cifar10
python Scripts/script.py --execution_id "2024-02-26-10-35-43" --cuda --model resnet18 --train_dataset cifar10  --cs_dataset fashion_mnist --l2  --desc "load cifar10 resnet18, compute cos_sim_adj and L2 on cifar10"

#load resnet18 cifar10, compute adj_cos sim and L2 on cifar100
python Scripts/script.py --execution_id "2024-02-26-10-35-43" --cuda --model resnet18 --train_dataset cifar10  --l2  --cs_dataset cifar100 --desc "load cifar10 resnet18, compute cos_sim_adj and L2 on cifar100"

#load resnet18 cifar100, compute adj_cos sim and L2 on cifar100
python Scripts/script.py --execution_id "2024-02-26-09-03-16" --cuda --model resnet18 --train_dataset cifar100 --cs_dataset cifar100  --l2  --desc "load cifar100 resnet18, compute cos_sim_adj and L2 on cifar100"

#load resnet18 cifar100, compute adj_cos sim and L2 on cifar10
python Scripts/script.py --execution_id "2024-02-26-09-03-16" --cuda --model resnet18 --train_dataset cifar100  --l2  --cs_dataset cifar10 --desc "load cifar100 resnet18, compute cos_sim_adj and L2 on cifar10"


echo "All the scripts have been executed"