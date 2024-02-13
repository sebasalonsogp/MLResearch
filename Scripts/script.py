import os, sys
sys.path.append('./models')
sys.path.append('../MLResearch')
#os.chdir('../MLResearch')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Scripts import *
from models import *
from resnet import *
from densenet import *
from datetime import datetime,date
from functions import *
import argparse
import json


def main():
    print("Running main function...")
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Specify the model')
    parser.add_argument('--test', action='store_true',default=False, help='Test the model')
    parser.add_argument('--cos_sim', action='store_true', default=False,  help='Calculate cosine similarity')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load a trained model')
    parser.add_argument('--nclass', type=int, required=False, help='Number of classes in the dataset')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('--train_dataset', type=str,required=False, help='Specify the training dataset')
    parser.add_argument('--evaluate_dataset', type=str, required=False, help='Specify the dataset for which the model was trained')
    parser.add_argument('--test_dataset', type=str, required=False, help='Specify the test dataset')
    parser.add_argument('--test_model', type=str,required=False, help='Which trained model to load and evaluate')
    parser.add_argument('--num_epochs', type=int, default=0,required=False, help='Number of epochs model will/was trained')

    parser.add_argument('--desc', type=str, required=False, help='Description of the experiment')
    parser.add_argument('--execution_id', type=str, required=False, help='Execution ID of model to load')
    args = parser.parse_args()
    
    execution_id = args.execution_id if args.execution_id else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    os.makedirs(f'./results/{execution_id}', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print("Device: ", device)

    data = {
        'vectors': [],
        'cosine_similarity': None,
        'Execution_ID': execution_id
    }

    if args.train:

        model =  get_model(args).to(device) 
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        cost = nn.CrossEntropyLoss()
        train_dataset = get_dataset(args.train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        torch.save(train(model, train_loader, cost, optimizer, args.num_epochs, device), f'./results/{execution_id}/model_{args.model}_{args.train_dataset}_epoch_{args.num_epochs}.pth')

    if args.test:

        model =  get_model(args)
        model.load_state_dict(torch.load(f'./results/{execution_id}/model_{args.model}_{args.evaluate_dataset}_epoch_{args.num_epochs}.pth'))
        model.to(device)
        model.eval()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        cost = nn.CrossEntropyLoss()

        test_dataset = get_dataset(args.test_dataset)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        data['vectors'].append(eval(model, test_loader, args.test_model, args.num_epochs,device, execution_id))

    if args.cos_sim:
        model = get_model(args)
        model.load_state_dict(torch.load(f'./results/{execution_id}/model_{args.model}_{args.evaluate_dataset}_epoch_{args.num_epochs}.pth'))
        #model = torch.load(f'./results/{execution_id}/model_{args.model}_{args.evaluate_dataset}_epoch_{args.num_epochs}.pth')
        model.to(device)
        model.eval()

        cos_sim_dataset = get_dataset(args.evaluate_dataset)
        cos_sim_loader = torch.utils.data.DataLoader(dataset=cos_sim_dataset, batch_size=64, shuffle=False)

        data['cosine_similarity'] = cos_sim(model,cos_sim_loader, device)    

    save_results(data, execution_id, args.desc)
    return execution_id


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
        #raise ValueError('Model not implemented yet')
        model = DenseNetCifar(args.nclass, scale=32, channels=3, proto_layer=4, layer_norm = False, entry_stride = 1)
    else:
        raise ValueError('Unrecognized model not implemented yet')
    return model

def get_dataset(args):
    if args == 'cifar10':
        dataset = datasets.CIFAR10(root='./Notebooks/data', train=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                                                                      transforms.Normalize((.5071,.4865,.4409), (.2675,.2565,.2761))]), download=True)
    elif args == 'cifar100':
        dataset = datasets.CIFAR100(root='./Notebooks/data', train=True, transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                                                       transforms.RandomHorizontalFlip(),
                                                                                                       transforms.ToTensor(),
                                                                                                       transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))]),
                                                                                                       download=True)
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


def save_results(data, execution_id, description=None):
    if data:
        with open(f'./results/{execution_id}/results.json', 'a') as file:
            if file.tell():
                file.write(",\n")
            json.dump(data, file)
            
    with open(f'./results/{execution_id}/info.txt', 'a') as file:
        file.write(f'Performing new computation on {datetime.now()}\n')
        file.write(f'Execution ID: {execution_id}\n')
        if description:
            file.write(f'Description: {description}\n')
        else:
            file.write(f'Description: No description provided\n')
        file.write(f'Finished computing on: {datetime.now()}\n\n')

    print("Results saved")



if __name__ == '__main__':
    main()

