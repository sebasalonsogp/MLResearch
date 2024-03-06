#Evaluate resnet18 mnist on mnist test set
python Scripts/script.py --test --test_dataset mnist --train_dataset mnist --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 mnist on fashion mnist test set
python Scripts/script.py --test --test_dataset fashion_mnist --train_dataset mnist --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 fashion  on fashion
python Scripts/script.py --test --test_dataset fashion_mnist --train_dataset fashion_mnist --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 fashion on mnist
python Scripts/script.py --test --test_dataset mnist --train_dataset fashion_mnist --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 cifar10 on cifar10
python Scripts/script.py --test --test_dataset cifar10 --train_dataset cifar10 --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 cifar10 on cifar100
python Scripts/script.py --test --test_dataset cifar100 --train_dataset cifar10 --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 cifar100 on cifar100
python Scripts/script.py --test --test_dataset cifar100 --train_dataset cifar100 --model resnet18 --desc "get test accuracy" --cuda

#evaluate resnet18 cifar100 on cifar10
python Scripts/script.py --test --test_dataset cifar10 --train_dataset cifar100 --model resnet18 --desc "get test accuracy" --cuda

#evaluate densenet cifar10 on cifar10
python Scripts/script.py --test --test_dataset cifar10 --train_dataset cifar10 --model densenet --desc "get test accuracy" --cuda

#evaluate densenet cifar10 on cifar100
python Scripts/script.py --test --test_dataset cifar100 --train_dataset cifar10 --model densenet --desc "get test accuracy" --cuda

#evaluate densenet cifar100 on cifar100
python Scripts/script.py --test --test_dataset cifar100 --train_dataset cifar100 --model densenet --desc "get test accuracy" --cuda

#evaluate densenet cifar100 on cifar10
python Scripts/script.py --test --test_dataset cifar10 --train_dataset cifar100 --model densenet --desc "get test accuracy" --cuda
