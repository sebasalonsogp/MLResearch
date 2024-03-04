#!/bin/bash
echo "Executing the cross-cos_sim scripts"

#load resnet18 mnist, compute cos_sim on fashion_mnist
python Scripts/script.py --execution_id "2024-03-03-12-18-11" --cuda --model resnet18 --train_dataset mnist  --cos_sim --cs_dataset fashion_mnist --desc "load mnist resnet18, compute cos_sim on fashion_mnist" --cos_sim_adj --l2

#load fashion_mnist resnet18, compute cos_sim on mnist
python Scripts/script.py --execution_id "2024-03-03-12-35-59" --cuda --model resnet18 --train_dataset fashion_mnist  --cos_sim --cs_dataset mnist --desc "load fashion_mnist resnet18, compute cos_sim on mnist" --cos_sim_adj --l2

#load cifar10 resnet18, compute cos_sim on cifar100
python Scripts/script.py --execution_id "2024-03-03-12-56-23" --cuda --model resnet18 --train_dataset cifar10  --cos_sim --cs_dataset cifar100 --desc "load cifar10 resnet18, compute cos_sim on cifar100" --cos_sim_adj --l2

#load cifar100 resnet18, compute cos_sim on cifar10
python Scripts/script.py --execution_id "2024-02-26-09-03-16" --cuda --model resnet18 --train_dataset cifar100  --cos_sim --cs_dataset cifar10 --desc "load cifar100 resnet18, compute cos_sim on cifar10" --cos_sim_adj --l2

#load cifar10 densenet121, compute cos_sim on cifar100
python Scripts/script.py --execution_id "2024-02-26-09-11-59" --cuda --model densenet --train_dataset cifar10  --cos_sim --cs_dataset cifar100 --desc "load cifar10 densenet121, compute cos_sim on cifar100" --cos_sim_adj --l2

#load cifar100 densenet121, compute cos_sim on cifar10
python Scripts/script.py --execution_id "2024-02-26-09-20-42" --cuda --model densenet --train_dataset cifar100  --cos_sim --cs_dataset cifar10 --desc "load cifar100 densenet121, compute cos_sim on cifar10" --cos_sim_adj --l2

echo "All the scripts have been executed"