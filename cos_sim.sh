#!/bin/bash
echo "Executing the cross-cos_sim scripts"

#load resnet18 mnist, compute cos_sim on fashion_mnist
python Scripts/script.py --execution_id "2024-02-26-08-25-11" --cuda --model resnet18 --train_dataset mnist  --cos_sim --cs_dataset fashion_mnist --desc "load mnist resnet18, compute cos_sim on fashion_mnist"

#load fashion_mnist resnet18, compute cos_sim on mnist
python Scripts/script.py --execution_id "2024-02-26-08-33-57" --cuda --model resnet18 --train_dataset fashion_mnist  --cos_sim --cs_dataset mnist --desc "load fashion_mnist resnet18, compute cos_sim on mnist"

#load cifar10 resnet18, compute cos_sim on cifar100
python Scripts/script.py --execution_id "2024-02-26-10-35-43" --cuda --model resnet18 --train_dataset cifar10  --cos_sim --cs_dataset cifar100 --desc "load cifar10 resnet18, compute cos_sim on cifar100"

#load cifar100 resnet18, compute cos_sim on cifar10
python Scripts/script.py --execution_id "2024-02-26-09-03-16" --cuda --model resnet18 --train_dataset cifar100  --cos_sim --cs_dataset cifar10 --desc "load cifar100 resnet18, compute cos_sim on cifar10"

echo "All the scripts have been executed"