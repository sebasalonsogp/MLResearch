#!/bin/bash
echo "Executing the training scripts"

#train resnet18 on mnist
python Scripts/script.py --cuda --model resnet18 --train --train_dataset mnist --test --test_dataset mnist --cos_sim --cs_dataset mnist --num_epochs 10 --desc "train resnet18 on mnist" --l2 --cos_sim_adj

#train resnet18 on fashion_mnist
python Scripts/script.py --cuda --model resnet18 --train --train_dataset fashion_mnist --test --test_dataset fashion_mnist --cos_sim --cs_dataset fashion_mnist --num_epochs 20 --desc "train resnet18 on fashion_mnist" --l2 --cos_sim_adj


#train resnet18 on cifar10
python Scripts/script.py --cuda --model resnet18 --train --train_dataset cifar10 --test --test_dataset cifar10 --cos_sim --cs_dataset cifar10 --num_epochs 90 --desc "train resnet18 on cifar10" --l2 --cos_sim_adj


#train resnet18 on cifar100
python Scripts/script.py --cuda --model resnet18 --train --train_dataset cifar100 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --num_epochs 170 --desc "train resnet18 on cifar100" --l2 --cos_sim_adj


#train densenet on mnist
python Scripts/script.py --cuda --model densenet --train --train_dataset cifar10 --test --test_dataset cifar10 --cos_sim --cs_dataset cifar10 --num_epochs 100 --desc "train densenet on cifar10" --l2 --cos_sim_adj

#train densenet on fashion_mnist
python Scripts/script.py --cuda --model densenet --train --train_dataset cifar100 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --num_epochs 200 --desc "train densenet on cifar100" --l2 --cos_sim_adj

echo "All the scripts have been executed"