#!/bin/bash
ed1=$(python Scripts/script.py --cuda --model resnet18 --train --train_dataset mnist --test --test_dataset mnist --cos_sim --cs_dataset mnist --num_epochs 10 --desc "train on mnist")
python Scripts/script.py --execution_id "$ed1" --cuda --model resnet18 --train_dataset mnist  --cos_sim --cs_dataset fashion_mnist --desc "load mnist resnet18, compute cos_sim on fashion_mnist"

ed2=$(python Scripts/script.py --cuda --model resnet18 --train --train_dataset fashion_mnist --test --test_dataset fashion_mnist --cos_sim --cs_dataset fashion_mnist --num_epochs 20 --desc "train on fashion_mnist")
python Scripts/script.py --execution_id "$ed2" --cuda --model resnet18 --train_dataset fashion_mnist  --cos_sim --cs_dataset mnist --desc "load fashion_mnist resnet18, compute cos_sim on mnist"


ed3=$(python Scripts/script.py --cuda --model resnet18 --train --train_dataset cifar10 --test --test_dataset cifar10 --cos_sim --cs_dataset cifar10 --num_epochs 30 --desc "train on cifar10")
python Scripts/script.py --execution_id "$ed3" --cuda --model resnet18 --train_dataset cifar10  --cos_sim --cs_dataset cifar100 --desc "load cifar10 resnet18, compute cos_sim on cifar100"

ed4=$(python Scripts/script.py --cuda --model resnet18 --train --train_dataset cifar100 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --num_epochs 150 --desc "train on cifar100")
python Scripts/script.py --execution_id "$ed4" --cuda --model resnet18 --train_dataset cifar100  --cos_sim --cs_dataset cifar10 --desc "load cifar100 resnet18, compute cos_sim on cifar10"

ed7=$(python Scripts/script.py --cuda --model densenet --train --train_dataset mnist --test --test_dataset mnist --cos_sim --cs_dataset mnist --num_epochs 10 --desc "train on cifar10")
python Scripts/script.py --execution_id "$ed7" --cuda --model densenet --train_dataset mnist  --cos_sim --cs_dataset fashion_mnist --desc "load mnist densenet, compute cos_sim on fashion_mnist"

ed8=$(python Scripts/script.py --cuda --model densenet --train --train_dataset fashion_mnist --test --test_dataset fashion_mnist --cos_sim --cs_dataset fashion_mnist --num_epochs 20 --desc "train on fashion_mnist")
python Scripts/script.py --execution_id "$ed8" --cuda --model densenet --train_dataset fashion_mnist  --cos_sim --cs_dataset mnist --desc "load fashion_mnist densenet, compute cos_sim on mnist"

ed5=$(python Scripts/script.py --cuda --model densenet --train --train_dataset cifar10 --test --test_dataset cifar10 --cos_sim --cs_dataset cifar10 --num_epochs 30 --desc "train on cifar10")
python Scripts/script.py --execution_id "$ed5" --cuda --model densenet --train_dataset cifar10  --cos_sim --cs_dataset cifar100 --desc "load cifar10 densenet, compute cos_sim on cifar100"

ed6=$(python Scripts/script.py --cuda --model densenet --train --train_dataset cifar100 --test --test_dataset cifar100 --cos_sim --cs_dataset cifar100 --num_epochs 150 --desc "train on cifar100")
python Scripts/script.py --execution_id "$ed6" --cuda --model densenet --train_dataset cifar100  --cos_sim --cs_dataset cifar10 --desc "load cifar100 densenet, compute cos_sim on cifar10"

