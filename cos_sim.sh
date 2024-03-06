#!/bin/bash
echo "Executing the cross-cos_sim scripts"

#load resnet18 mnist, compute cos_sim on fashion_mnist
python Scripts/script.py --cuda --model resnet18 --train_dataset mnist --test --test_dataset fashion_mnist --cos_sim --cs_dataset fashion_mnist --desc "load mnist resnet18, compute cos_sim on fashion_mnist" --cos_sim_adj 

#load fashion_mnist resnet18, compute cos_sim on mnist
python Scripts/script.py --cuda --model resnet18 --train_dataset fashion_mnist --test --test_dataset mnist  --cos_sim --cs_dataset mnist --desc "load fashion_mnist resnet18, compute cos_sim on mnist" --cos_sim_adj 

#load cifar10 resnet18, compute cos_sim on cifar100
python Scripts/script.py  --cuda --model resnet18 --train_dataset cifar10 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --desc "load cifar10 resnet18, compute cos_sim on cifar100" --cos_sim_adj

#load cifar100 resnet18, compute cos_sim on cifar10
python Scripts/script.py  --cuda --model resnet18 --train_dataset cifar100 --test --test_dataset cifar10 --cos_sim --cs_dataset cifar10 --desc "load cifar100 resnet18, compute cos_sim on cifar10" --cos_sim_adj

#load cifar10 densenet, compute cos_sim on cifar100
python Scripts/script.py --cuda --model densenet --train_dataset cifar10 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --desc "load cifar10 densenet121, compute cos_sim on cifar100" --cos_sim_adj

#load cifar100 densenet, compute cos_sim on cifar10
python Scripts/script.py  --cuda --model densenet --train_dataset cifar100 --test --test_dataset cifar10  --cos_sim --cs_dataset cifar10 --desc "load cifar100 densenet121, compute cos_sim on cifar10" --cos_sim_adj

echo "All the scripts have been executed"