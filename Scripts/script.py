import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from Scripts import *
from data import *
from models import *
from datetime import datetime
from functions import *
import pickle
import argparse
import json
import os

def main():

    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(f'../data/{execution_id}', exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Specify the model')
    parser.add_argument('--nclass', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('--train_dataset', type=str,required=False, help='Specify the training dataset')
    parser.add_argument('--cos_sim', action='store_true', default=False, help='Calculate cosine similarity')
    parser.add_argument('--trained_dataset', type=str, required=True, help='Specify the dataset for which the model was trained')
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    if args == 'resnet18':
        model = ResNet18(args.nclass, scale=64, channels=1, proto_layer=4,layer_norm = False, entry_stride = 1)
    elif args == 'resnet34':
        raise ValueError('Model not implemented yet')
        #model = ResNet34()
    elif args == 'resnet50':
        raise ValueError('Model not implemented yet')
        #model = ResNet50()
    elif args == 'resnet101':
        raise ValueError('Model not implemented yet')
        #model = ResNet101()
    elif args == 'resnet152':
        raise ValueError('Model not implemented yet')
        #model = ResNet152()
    elif args == 'densenet':
        raise ValueError('Model not implemented yet')
        #model = DenseNet()
    else:
        raise ValueError('Model not implemented yet')
    return model

def get_dataset(args):
    if args == 'cifar10':
        dataset = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
    elif args == 'cifar100':
        dataset = datasets.CIFAR100(root='data', train=True, transform=transforms.ToTensor(), download=True)
    elif args == 'mnist':
        dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    elif args == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError('Dataset not recognized')
    return dataset


##TODO add args for optimizer and loss function


def save_results(data, execution_id, train, train_dataset=None, num_epochs=None):

    if train:
        torch.save(data['trained_network'], f'../data/{execution_id}/model_{train_dataset}_epoch_{num_epochs+1}.pth')

    del data['trained_network']
    
    with open(f'../data/{execution_id}/results.json', 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    main()
