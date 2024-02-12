import os, sys
sys.path.append('./models')
sys.path.append('../MLResearch')
#os.chdir('../MLResearch')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from Scripts import *
from models import *
from resnet import *
from datetime import datetime
from functions import *
import argparse
import json


def main():
    print("Running main function...")
    execution_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


    os.makedirs(f'./results/{execution_id}', exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Specify the model')
    parser.add_argument('--nclass', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('--train_dataset', type=str,required=False, help='Specify the training dataset')
    parser.add_argument('--cos_sim', action='store_true', default=False,  help='Calculate cosine similarity')
    parser.add_argument('--trained_dataset', type=str, required=False, help='Specify the dataset for which the model was trained')
    parser.add_argument('--test', action='store_true',default=False, help='Test the model')
    parser.add_argument('--test_dataset', type=str, required=False, help='Specify the test dataset')
    parser.add_argument('--test_model', type=str,required=False, help='Which trained model to load and evaluate')
    parser.add_argument('--num_epochs', type=int, default=0,required=False, help='Number of epochs to train the model')

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    model = get_model(args).to(device)

    data = {
        'trained_network': model, 
        'vectors': [],
        'cosine_similarity': None
    }

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    cost = nn.CrossEntropyLoss()

    if args.train:
        train_dataset = get_dataset(args.train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        data['trained_network'] = train(model, train_loader, cost, optimizer, args.num_epochs, device)

    if args.test:
        test_dataset = get_dataset(args.test_dataset)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        data['vectors'].append(eval(model, test_loader, args.test_model, args.num_epochs,device, execution_id))

    if args.cos_sim:
        cos_sim_dataset = get_dataset(args.trained_dataset)
        cos_sim_loader = torch.utils.data.DataLoader(dataset=cos_sim_dataset, batch_size=64, shuffle=False)
        data['cosine_similarity'] = cos_sim(model,cos_sim_loader, args.trained_model, args.num_epochs,device, execution_id)    

    save_results(data, execution_id, args.train, args.train_dataset, args.num_epochs)
   
    return data


def get_model(args):
 
    if args.model == 'resnet18':
        model = ResNet18(args.nclass, scale=64, channels=1, proto_layer=4,layer_norm = False, entry_stride = 1)
    elif args.model == 'resnet34':
        raise ValueError('Model not implemented yet')
        #model = ResNet34()
    elif args.model == 'resnet50':
        raise ValueError('Model not implemented yet')
        #model = ResNet50()
    elif args.model == 'resnet101':
        raise ValueError('Model not implemented yet')
        #model = ResNet101()
    elif args.model == 'resnet152':
        raise ValueError('Model not implemented yet')
        #model = ResNet152()
    elif args.model == 'densenet':
        raise ValueError('Model not implemented yet')
        #model = DenseNet()
    else:
        raise ValueError('Unrecognized model not implemented yet')
    return model

def get_dataset(args):
    if args == 'cifar10':
        dataset = datasets.CIFAR10(root='./Notebooks/data', train=True, transform=transforms.ToTensor(), download=True)
    elif args == 'cifar100':
        dataset = datasets.CIFAR100(root='./Notebooks/data', train=True, transform=transforms.ToTensor(), download=True)
    elif args == 'mnist':

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./Notebooks/data', train=True, transform=transform, download=True)
    elif args == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root='./Notebooks/data', train=True, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError('Dataset not recognized')
    return dataset


##TODO add args for optimizer and loss function


def save_results(data, execution_id, train=False, train_dataset=None, num_epochs=None):
    if train:
        print("train=True")
        torch.save(data['trained_network'], f'./results/{execution_id}/model_{train_dataset}_epoch_{num_epochs}.pth')

    del data['trained_network']
    with open(f'./results/{execution_id}/results.json', 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    main()

