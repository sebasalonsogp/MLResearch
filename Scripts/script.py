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
    parser.add_argument('--load_model_train', action='store_true', default=False, help='Load model and continue train?')
    

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


    if args.execution_id: ##NOTE: Only use if u wish to recompute something otherwise it will overwrite data!
        try:
            with open(f'{model_path}/{args.execution_id}/results.json', 'r') as file:
                data = json.load(file)
        except FileNotFoundError as e:
            logging.error(f"Failed to load results. Check if execution_id is correct. Error: {e}")
            raise e("Results not found")
    else:
        data = {
        'Execution_ID': execution_id,
        'Start Time': start_time,
        'End Time': None,
        'Elapsed Time': None,
        'Training Accuracy': None,
        'Test Accuracy': None,
        'cosine_similarity': None,
        'adj_cos_sim': None,
        'L2_distance': None
    }

   

    training_settings = None

    nclass,nchannels = get_model_params(args.train_dataset)

    model = get_model(args, nclass,nchannels)
    

    ##TODO adjust code such that train function automatically computes training accuracy and test function automatically uses test accuracy


    test_accuracy, train_accuracy = None,None
    
    if args.train:
        try:
            os.makedirs(f'./results/{args.model}/{args.train_dataset}', exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create save path for model and dataset. Check if model and dataset are specified correctly. Error: {e}")
            raise ValueError("Failed to create save path for model and dataset. Check if model and dataset are specified correctly.")

        optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay=0)
        cost = nn.CrossEntropyLoss()

        try:
            train_dataset, batch_size = get_dataset(args.train_dataset,eval_train=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            
            test_dataset, batch_size = get_dataset(args.train_dataset,eval_train=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load training/test dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        try:
            if args.load_model_train:
                model.load_state_dict(
                    torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}.pth')
                    )
                print(f'Loaded model_{args.model}_{args.train_dataset}.pth')
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        
        computed_model, actual_epochs = train(model=model,test_loader=test_loader,train_loader=train_loader, cost=cost, optimizer=optimizer, num_epochs=args.num_epochs, device=device)
        train_accuracy = eval(model=computed_model, eval_dataloader=train_loader, device=device)
        args.num_epochs = actual_epochs


        training_settings = {'model': args.model, 'train_dataset': args.train_dataset, 'num_epochs': args.num_epochs, 'optimizer': 'SGD', 'loss_function': 'CrossEntropyLoss', 'learning_rate': 0.001, 'batch_size': batch_size, 'weight_decay': 0, 'entry_stride': 1, 'layer_norm': 'False', 'proto_layer': 4, 'scale': 64 if args.model=='resnet18' else 32, 'channels': '3' if args.train_dataset=='cifar10' or args.train_dataset=='cifar100' else '1'}
        with open(f'{result_path}/training_settings.json', 'w') as file:
            json.dump(training_settings, file)
        torch.save(computed_model, f'{model_path}/model_{args.model}_{args.train_dataset}.pth')

        
    
   

    if args.test:
        
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}.pth')
                )
        except ValueError as e: 
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        print(f'Loaded model_{args.model}_{args.train_dataset}.pth')
        try:
            test_dataset, batch_size = get_dataset(args.test_dataset,eval_train=False)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        except ValueError as e:
            logging.log(f"Failed to load test dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")

        print(f"Testing model on {args.test_dataset} dataset...")
        test_accuracy = eval(model=model, eval_dataloader=test_loader, device=device)

    if args.cos_sim:
    
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        print(f'Loaded model_{args.model}_{args.train_dataset}.pth')
        try:
            cos_sim_dataset, batch_size = get_dataset(args.cs_dataset,eval_train=True)
            cos_sim_loader = torch.utils.data.DataLoader(dataset=cos_sim_dataset, batch_size=batch_size, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load cos_sim dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")

        data['cosine_similarity'] = cos_sim(model=model,cs_dataloader=cos_sim_loader, device=device)    


    if args.cos_sim_adj:
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        print(f'Loaded model_{args.model}_{args.train_dataset}.pth')
        try:
            adj_cos_sim_dataset, batch_size = get_dataset(args.cs_dataset)
            adj_cos_sim_loader = torch.utils.data.DataLoader(dataset=adj_cos_sim_dataset, batch_size=batch_size, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load adj_cos_sim dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        data['adj_cos_sim'] = cos_sim_adj(model=model,cs_dataloader=adj_cos_sim_loader, device=device)

    if args.l2:
        try:
            model.load_state_dict(
                torch.load(f'{model_path}/model_{args.model}_{args.train_dataset}.pth')
                )
        except ValueError as e:
            logging.error(f"Failed to load model. Check if model is specified correctly. Error: {e}")
            raise e("Model not found")
        print(f'Loaded model_{args.model}_{args.train_dataset}.pth')
        try:
            l2_dataset, batch_size = get_dataset(args.cs_dataset)
            l2_loader = torch.utils.data.DataLoader(dataset=l2_dataset, batch_size=batch_size, shuffle=False)
        except ValueError as e:
            logging.error(f"Failed to load l2 dataset. Check if dataset is specified correctly. Error: {e}")
            raise e("Dataset not found")
        
        data['L2_distance'] = l2_dist(model=model,cs_dataloader=l2_loader, device=device)

    

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = str(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    data['End Time'], data['Elapsed Time'] = end_time, elapsed_time

    if args.train:
        data['Training Accuracy'] = train_accuracy
    if args.test:
        data['Test Accuracy'] = test_accuracy

    save_results(data, args, training_settings=training_settings,test_accuracy=test_accuracy,train_accuracy=train_accuracy,elapsed_time=elapsed_time, result_path=result_path, execution_id=execution_id)

    print(f"Finished running script at {end_time}.\nTotal time elapsed: {elapsed_time}")

    
    return execution_id


def save_results(data, args,training_settings=None, elapsed_time=None,result_path=None, execution_id=None, test_accuracy=None,train_accuracy=None):
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
                formatted_training_settings = json.dumps(training_settings, indent=4)
                file.write(f"Training settings: {formatted_training_settings}\n")
                file.write(f"Saved Model: model_{args.model}_{args.train_dataset}\n")
                file.write(f'Training accuracy: {train_accuracy}\n' if train_accuracy else "No training accuracy computed.\n")
            if args.test:
                file.write(f"Tested model on dataset {args.test_dataset}.\n")
                file.write(f'Test accuracy: {test_accuracy}\n' if test_accuracy else "No test accuracy computed.\n")
            if args.cos_sim:
                file.write(f"Calculated cosine similarity on dataset {args.cs_dataset}.\n")
            if args.cos_sim_adj:
                file.write(f"Calculated mean-adjusted cosine similarity on dataset {args.cs_dataset}.\n")
            if args.l2:
                file.write(f"Calculated L2 distance on dataset {args.cs_dataset}.\n")
            
        file.write(f'\nTotal elapsed time: {elapsed_time}')
        
        file.write(f'\n------------------------------------------------ Finished computing on: {datetime.now()} -----------------------------------------------------------------\n\n')

    print("Results saved")

    return None

def get_model(args,nclass,channels):
    if args.model == 'resnet18':
        model = ResNet18(nclass, scale=64, channels=channels, proto_layer=4,layer_norm = False, entry_stride = 1)
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
        model = DenseNetCifar(nclass, scale=32, channels=channels, proto_layer=4, layer_norm = False, entry_stride = 1)
    else:
        raise ValueError('Unrecognized model not implemented yet')
    return model

def get_dataset(args,eval_train=False):
    batch_size = None
    if args == 'cifar10':
        batch_size = 128
        dataset = datasets.CIFAR10(root='./Notebooks/data', train=eval_train, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                                                                                      transforms.Normalize((.5071,.4865,.4409), (.2675,.2565,.2761))]), download=True)
    elif args == 'cifar100':
        batch_size = 128
        dataset = datasets.CIFAR100(root='./Notebooks/data', train=eval_train, transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                                                       transforms.RandomHorizontalFlip(),
                                                                                                       transforms.ToTensor(),
                                                                                                       transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))]),
                                                                                                       download=True)
    elif args == 'mnist':
        batch_size = 64
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./Notebooks/data', train=eval_train, transform=transform, download=True)
    elif args == 'fashion_mnist':
        batch_size = 64
        dataset = datasets.FashionMNIST(root='./Notebooks/data', train=eval_train, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError('Dataset not recognized')
    return dataset, batch_size


def get_model_params(arg=None):
    if arg == 'mnist' or arg == 'fashion_mnist':
        nclass = 10
        nchannels = 1
    elif arg == 'cifar10':
        nclass = 10
        nchannels = 3
    elif arg == 'cifar100':
        nclass = 100
        nchannels = 3
    else:
        raise ValueError("Dataset not recognized, couldnt determine number of classes or channels. Please specify nclass or nchannels.")
    return nclass, nchannels

##TODO add args for optimizer and loss function

if __name__ == '__main__':
    main()