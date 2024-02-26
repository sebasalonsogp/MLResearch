import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Scripts.functions import train, cos_sim, eval,cos_sim_adj, l2_dist
from models.resnet import *
from models.densenet import *
from datetime import datetime
import argparse, json, logging
from datetime import datetime


def main():
    ##TODO write test function
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Specify the model.')
    parser.add_argument('--test', action='store_true',default=False, help='Test the model?')
    parser.add_argument('--test_dataset', type=str, required=False, help='Specify the evaluation dataset (eval or cos_sim)')
    parser.add_argument('--cos_sim', action='store_true', default=False,  help='Calculate cosine similarity?')
    parser.add_argument('--cs_dataset', type=str, required=False, help='Specify the dataset for which to compute cosine similarity.')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model?')
    parser.add_argument('--train_dataset', type=str,required=True, help='Specify the dataset to train or was trained on.')
    parser.add_argument('--num_epochs', type=int, default=None,required=False, help='Number of epochs model will/was trained')
    parser.add_argument('--desc', type=str, required=False, help='Description of the experiment')
    parser.add_argument('--execution_id', type=str, required=False, help='Execution ID of model to load')
    parser.add_argument('--cuda', action='store_true', default=False, help='Require cuda?')
    parser.add_argument('--l2', action='store_true', default=False, help='Compute L2 distance?')
    parser.add_argument('--cos_sim_adj', action='store_true', default=False, help='Compute mean-adjusted cosine similarity?')

    args = parser.parse_args()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started running script at {start_time} ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Device: {device}")

    if args.cuda and device == 'cpu':
        raise ValueError("CUDA is not available")

    execution_id = args.execution_id if args.execution_id else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    model_path = f'./results/{args.model}/{args.train_dataset}'
    result_path = f'{model_path}/{execution_id}'
    os.makedirs(result_path, exist_ok=True)

    logging.basicConfig(filename=f'{result_path}/error.log', level=logging.ERROR,format='%(asctime)s:%(levelname)s:%(message)s')
    
    data = {
        'Execution_ID': execution_id,
        'Start Time': start_time,
        'End Time': None,
        'Elapsed Time': None,
        'cosine_similarity': None,
        'adj_cos_sim': None,
        'L2_distance': None
    }

    nclass = None
    if args.train_dataset:
        if args.train_dataset == 'cifar10' or args.train_dataset == 'mnist' or args.train_dataset == 'fashion_mnist':
            nclass = 10
        elif args.train_dataset == 'cifar100':
            nclass = 100
        else:
            raise ValueError("Dataset not recognized, couldnt determine number of classes. Please specify nclass.")
    model = get_model(args, nclass)
    

    if args.train:
        try:
            os.makedirs(f'./results/{args.model}/{args.train_dataset}', exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create save path for model and dataset. Check if model and dataset are specified correctly. Error: {e}")
            raise ValueError("Failed to create save path for model and dataset. Check if model and dataset are specified correctly.")

        optimizer = optim.SGD(model.parameters(), lr=0.001)
        cost = nn.CrossEntropyLoss()

        try:
            train_dataset = get_dataset(args.train_dataset)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        except ValueError as e:
            logging.error(f"Failed to load training dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        computed_model, actual_epochs = train(model=model, train_loader=train_loader, cost=cost, optimizer=optimizer, num_epochs=args.num_epochs, device=device)
        args.num_epochs = actual_epochs

        torch.save(computed_model, f'{model_path}/model_{args.model}_{args.train_dataset}_id_{execution_id}.pth')

    if args.test:
        
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}_id_{execution_id}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")

        try:
            test_dataset = get_dataset(args.test_dataset)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        except ValueError as e:
            logging.log(f"Failed to load test dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")

        accuracy = eval(model=model, eval_dataloader=test_loader, device=device)

    if args.cos_sim:
    
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}_id_{execution_id}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        
        try:
            cos_sim_dataset = get_dataset(args.cs_dataset)
            cos_sim_loader = torch.utils.data.DataLoader(dataset=cos_sim_dataset, batch_size=64, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load cos_sim dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")

        data['cosine_similarity'] = cos_sim(model=model,cs_dataloader=cos_sim_loader, device=device)    


    if args.cos_sim_adj:
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}_id_{execution_id}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        
        try:
            adj_cos_sim_dataset = get_dataset(args.cs_dataset)
            adj_cos_sim_loader = torch.utils.data.DataLoader(dataset=adj_cos_sim_dataset, batch_size=64, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load adj_cos_sim dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        data['adj_cos_sim'] = cos_sim_adj(model=model,cs_dataloader=adj_cos_sim_loader, device=device)

    if args.l2:
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}_id_{execution_id}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        
        try:
            l2_dataset = get_dataset(args.cs_dataset)
            l2_loader = torch.utils.data.DataLoader(dataset=l2_dataset, batch_size=64, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load l2 dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        data['L2_distance'] = l2_dist(model=model,cs_dataloader=l2_loader, device=device)

    

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    data['End Time'], data['Elapsed Time'] = end_time, str(elapsed_time)

    save_results(data, args, result_path, execution_id)

    print(f"Finished running script at {end_time}.\nTotal time elapsed: {elapsed_time}")

    
    return execution_id



def save_results(data, args, result_path, execution_id):
    print("Saving results...")
    
    if data:
        with open(f'{result_path}/results.json', 'a') as file:
            if file.tell():
                file.write(",\n")
            json.dump(data, file)
            
    with open(f'{result_path}/info.txt', 'a') as file:
        
        if file.tell():
                file.write("\n")

        file.write(f'------------------------------------------------ Performing new computation on {datetime.now()} --------------------------------------------------------\n')
        file.write(f'\nExecution ID: {execution_id}')
        
        if args.desc:
            file.write(f'\nDescription: {args.desc}\n\n')
        else:
            file.write(f'\nDescription: No description provided\n\n')

        if args:
            args_dict = vars(args)
            for arg in args_dict:
                file.write(f'{arg}: {args_dict[arg]}\n')
        
        if args.model and args.train_dataset:
            file.write(f"\nUsed model {args.model} with dataset {args.train_dataset}.\n")
            if args.train:
                file.write(f"Trained model for {args.num_epochs} epochs.\n")
                file.write(f"Saved Model: model_{args.model}_{args.train_dataset}_id_{execution_id}\n")
            if args.test:
                file.write(f"Tested model on dataset {args.test_dataset}.\n")
            if args.cos_sim:
                file.write(f"Calculated cosine similarity on dataset {args.cs_dataset}.\n")
            if args.cos_sim_adj:
                file.write(f"Calculated mean-adjusted cosine similarity on dataset {args.cs_dataset}.\n")
            if args.l2:
                file.write(f"Calculated L2 distance on dataset {args.cs_dataset}.\n")
            
        
        file.write(f'\n------------------------------------------------ Finished computing on: {datetime.now()} -----------------------------------------------------------------\n\n')

    print("Results saved")

    return None

def get_model(args,nclass):
    if args.model == 'resnet18':
        model = ResNet18(nclass, scale=64, channels=1, proto_layer=4,layer_norm = False, entry_stride = 1)
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
        model = DenseNetCifar(nclass, scale=32, channels=3, proto_layer=4, layer_norm = False, entry_stride = 1)
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

if __name__ == '__main__':
    main()