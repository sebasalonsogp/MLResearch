import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas,json,os


def train(model=None,train_loader=None,test_loader=None, cost=None, optimizer=None, num_epochs=None, device=None):

    if model is None:
        raise ValueError("Model is not defined")
    elif train_loader is None:
        raise ValueError("Dataset is not defined")
    elif test_loader is None:
        raise ValueError("Test dataset is not defined")
    elif cost is None:
        raise ValueError("Cost function is not defined")
    elif optimizer is None:
        raise ValueError("Optimizer is not defined")
    elif num_epochs is None:
        raise ValueError("Number of epochs is not defined")
    elif device is None:
        raise ValueError("Device is not defined")


    print("Training the model...")
    
    model = model.to(device)

    # total_step = len(train_loader)

    # lowest_loss = float('inf')
    best_model = None
    best_test_accuracy=0
    actual_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train()
        curr_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            
            

            images = images.to(device)
            labels = labels.to(device)

            ## forward pass ##
            _,outputs = model(images)
            loss = cost(outputs, labels)

            ## backwards pass and optimizer step (learning) ##
            optimizer.zero_grad()  # zeroes out the gradients, removing exisitng ones to avoid accumulation
            loss.backward()  # gradient of loss, how much each parameter contributed to the loss
            optimizer.step()  # adjusts parameters based on results to minimize loss
            
            curr_loss += loss.item()

        model.eval()
        with torch.no_grad():
            correct,total=0,0
            for images,labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy = 100 * correct / total
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model = model
    

            #if (i + 1) % 100 == 0:
                #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # avg_loss = curr_loss / len(train_loader)        
        # if avg_loss < lowest_loss:
        #     print(f"Loss decreased from {lowest_loss} to {avg_loss}. Saving the current best model...")
        #     lowest_loss = avg_loss
        #     actual_epoch = epoch
        #     best_model = model.state_dict()
        
    return best_model, actual_epoch+1 


def cos_sim(model=None, cs_dataloader=None, device=None):

    if model is None:
        raise ValueError("Model is not defined")
    elif cs_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")


    print("Computing cosine similarity...")

    model.to(device)
    model.eval()

    class_similarities = {}

    with torch.no_grad():
        for inputs, labels in cs_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            feat_vec, _ = model(inputs)  ## get feature vectors from the model

            for i in range(len(feat_vec)):
                for j in range(i+1, len(feat_vec)):

                    sim = F.cosine_similarity(feat_vec[i].unsqueeze(0), feat_vec[j].unsqueeze(0)) ## calculate cosine similarity between feature vectors

                    if labels[i] == labels[j]:   #within class
                        
                        class_id = labels[i].item()

                        if class_id not in class_similarities:
                            class_similarities[class_id] = {'intra_sim': [], 'inter_sim': []}

                        class_similarities[class_id]['intra_sim'].append(sim.item())
                    else: #between class

                        class_id_i, class_id_j = labels[i].item(), labels[j].item()

                        if class_id_i not in class_similarities:
                            class_similarities[class_id_i] = {'intra_sim': [], 'inter_sim': []}

                        if class_id_j not in class_similarities:
                            class_similarities[class_id_j] = {'intra_sim': [], 'inter_sim': []}
                            
                        class_similarities[class_id_i]['inter_sim'].append(sim.item())
                        class_similarities[class_id_j]['inter_sim'].append(sim.item())

    return class_similarities
    

def get_fv_μ(model=None, cs_dataloader=None, device=None):
    if model is None:
        raise ValueError("Model is not defined")
    elif cs_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")
    
    print("Computing adjusted cosine similarity...")

    model.to(device)
    model.eval()

    fv_sum ,cnt= None,0

    with torch.no_grad():
        for inputs, labels in cs_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            feat_vec, _ = model(inputs)

            if fv_sum is None:
                fv_sum = torch.zeros(feat_vec.shape[1],device=device)

            fv_sum += torch.sum(feat_vec, dim=0)
            #print(f'fv_sum {fv_sum.shape}')
            cnt+=feat_vec.size(0)

    μ = fv_sum/cnt
    return μ

def cos_sim_adj(model=None, cs_dataloader=None, device=None):

    if model is None:
        raise ValueError("Model is not defined")
    elif cs_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")
    μ = get_fv_μ(model, cs_dataloader, device)
    if μ is None:
        raise ValueError("μ is not defined")
    #print(f'Mu shape: {μ.shape}')

    print("Computing adjusted cosine similarity...")

    model.to(device)
    model.eval()

    class_similarities = {}

    with torch.no_grad():
        for inputs, labels in cs_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            feat_vec, _ = model(inputs)  ## get feature vectors from the model

            for i in range(len(feat_vec)):
                for j in range(i+1, len(feat_vec)):
                
                    sim = F.cosine_similarity(feat_vec[i].unsqueeze(0)-μ.unsqueeze(0), feat_vec[j].unsqueeze(0)-μ.unsqueeze(0)) ## calculate cosine similarity between feature vectors and convert it to a scalar
                    #print(f"Sim_Shape: {sim.shape}")

                    if labels[i] == labels[j]:   #within class
                        
                        class_id = labels[i].item()

                        if class_id not in class_similarities:
                            class_similarities[class_id] = {'intra_sim': [], 'inter_sim': []}

                        class_similarities[class_id]['intra_sim'].append(sim.item())
                    else: #between class

                        class_id_i, class_id_j = labels[i].item(), labels[j].item()

                        if class_id_i not in class_similarities:
                            class_similarities[class_id_i] = {'intra_sim': [], 'inter_sim': []}

                        if class_id_j not in class_similarities:
                            class_similarities[class_id_j] = {'intra_sim': [], 'inter_sim': []}
                            
                        class_similarities[class_id_i]['inter_sim'].append(sim.item())
                        class_similarities[class_id_j]['inter_sim'].append(sim.item())

    return class_similarities

def l2_dist(model=None, cs_dataloader=None, device=None):

    if model is None:
        raise ValueError("Model is not defined")
    elif cs_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")
    
    print("Computing L2 distance...")

    model.to(device)
    model.eval()

    class_distances = {}
    with torch.no_grad():
        for inputs, labels in cs_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            feat_vec, _ = model(inputs)  # get feature vectors from the model

            for i in range(len(feat_vec)):
                for j in range(i + 1, len(feat_vec)):

                    dist = torch.norm(feat_vec[i] - feat_vec[j], p=2)  # compute L2 distance

                    if labels[i] == labels[j]:  # within class
                        class_id = labels[i].item()
                        if class_id not in class_distances:
                            class_distances[class_id] = {'intra_dist': [], 'inter_dist': []}
                        class_distances[class_id]['intra_dist'].append(dist.item())

                    else:  # between class
                        class_id_i, class_id_j = labels[i].item(), labels[j].item()

                        if class_id_i not in class_distances:
                            class_distances[class_id_i] = {'intra_dist': [], 'inter_dist': []}

                        if class_id_j not in class_distances:
                            class_distances[class_id_j] = {'intra_dist': [], 'inter_dist': []}

                        class_distances[class_id_i]['inter_dist'].append(dist.item())
                        class_distances[class_id_j]['inter_dist'].append(dist.item())

    return class_distances

def eval(model=None, eval_dataloader=None, device=None):
        
    if model is None:
        raise ValueError("Model is not defined")
    elif eval_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")


    print(f"Evaluating the model...")

    
    model.to(device)
    model.eval()
    with torch.no_grad():

        feature_vectors, labels_list = [], []
        correct, total = 0,0
        for images, labels in eval_dataloader:

            images = images.to(device) #images
            labels = labels.to(device) #true labels
            features, outputs = model(images) #TODO which one returns 
            feature_vectors.append(features)  # Append the features to the list
            #labels_list.append(labels)  # Append the labels to the list

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        feature_vectors = torch.cat(feature_vectors, dim=0)  # Concatenate the feature vectors
        #labels_list = torch.cat(labels_list, dim=0)  # Concatenate the labels

        accuracy = 100 * correct / total
        print('Accuracy: {:.2f}%'.format(accuracy))
        print('Feature Vectors:', feature_vectors)
        #print('Labels:', labels_list)

        return accuracy
    

def plot_line_graph(cos_sim_matrix_np,title="Cosine Similarity Line Graph",sample_size=1000):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(title)

    for class_id in cos_sim_matrix_np:
        
        intra_similarities = np.random.choice(cos_sim_matrix_np[class_id]['intra_sim'], size=sample_size, replace=False)
        inter_similarities = np.random.choice(cos_sim_matrix_np[class_id]['inter_sim'], size=sample_size, replace=False)

        axs[0].plot(range(len(intra_similarities)), intra_similarities, label=f'class {class_id}')
        axs[1].plot(range(len(inter_similarities)), inter_similarities, label=f'class {class_id}')

    axs[0].set_title("Intra-class Similarity")
    axs[0].set_xlabel("Sample pair")
    axs[0].set_ylabel("Cosine similarity")

    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axs[1].set_title("Inter-class Similarity")
    axs[1].set_xlabel("Sample pair")
    axs[1].set_ylabel("Cosine similarity")
    
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


def plot_hist(cos_sim_matrix_np,title="Cosine Similarity Histogram"):

    for class_id in cos_sim_matrix_np:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Cosine Similarity for Class: {class_id}")

        # Intra-class similarity
        axs[0].hist(cos_sim_matrix_np[class_id]['intra_sim'], bins=50, color='blue', edgecolor='black', alpha=0.5, histtype='step')
        axs[0].set_title(f"Intra-class similarity")
        axs[0].set_xlabel("cosine similarity")
        axs[0].set_ylabel("Frequency")

        # Inter-class similarity
        axs[1].hist(cos_sim_matrix_np[class_id]['inter_sim'], bins=50, color='red', edgecolor='black', alpha=0.5,histtype='step')
        axs[1].set_title(f"Inter-class similarity")
        axs[1].set_xlabel("cosine similarity")
        axs[1].set_ylabel("Frequency")

        # Show the figure for each class_id
        plt.show()


def plot_scatter(cos_sim_matrix_np, sample_size=1000):
    for class_id in cos_sim_matrix_np:
        intra_sim_samples = np.random.choice(cos_sim_matrix_np[class_id]['intra_sim'], size=sample_size, replace=False)
        inter_sim_samples = np.random.choice(cos_sim_matrix_np[class_id]['inter_sim'], size=sample_size, replace=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(range(len(intra_sim_samples)), intra_sim_samples, color='blue', alpha=0.5, label='intra-class similarity')
        ax.scatter(range(len(inter_sim_samples)), inter_sim_samples, color='red', alpha=0.5, label='inter-class similarity')
        ax.set_title(f"Scatter plot of cosine similarity for Class: {class_id}")
        ax.set_xlabel("Sample pair")
        ax.set_ylabel("cosine similarity")
        ax.legend()
        plt.show()

def aggregated_hist(cos_sim_matrix_np,title="Aggregated Cosine Similarity Histogram",histtype='step'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    # Intra-class similarity
    #all_intra_sim = np.array([sim for class_id in cos_sim_matrix_np for sim in cos_sim_matrix_np[class_id]['intra_sim']])
    colors_intra = cm.get_cmap('Blues', len(cos_sim_matrix_np))
    for i, class_id in enumerate(cos_sim_matrix_np):
        axs[0].hist(cos_sim_matrix_np[class_id]['intra_sim'], bins=50, color=colors_intra(i), edgecolor=colors_intra(i), alpha=0.5, histtype=histtype)
    axs[0].set_title(f"Intra-class similarity")
    axs[0].set_xlabel("cosine similarity")
    axs[0].set_ylabel("Frequency")


    # Inter-class similarity
    #all_inter_sim = np.array([sim for class_id in cos_sim_matrix_np for sim in cos_sim_matrix_np[class_id]['inter_sim']])
    colors_inter = cm.get_cmap('Reds', len(cos_sim_matrix_np))
    for j, class_id in enumerate(cos_sim_matrix_np):
        axs[1].hist(cos_sim_matrix_np[class_id]['inter_sim'], bins=50, color=colors_inter(j), edgecolor=colors_inter(j), alpha=0.5, histtype=histtype)
    axs[1].set_title(f"Inter-class similarity")
    axs[1].set_xlabel("cosine similarity")
    axs[1].set_ylabel("Frequency")

    # Show the figure
    plt.show()


def get_stats_class(cos_sim_matrix_np):
    for class_id in cos_sim_matrix_np:
        print(f"Class {class_id}:\nIntra-class Similarity: Mean = {np.mean(cos_sim_matrix_np[class_id]['intra_sim'])}, Std = {np.std(cos_sim_matrix_np[class_id]['intra_sim'])}, Var = {np.var(cos_sim_matrix_np[class_id]['intra_sim'])}\nInter-class similarity: Mean = {np.mean(cos_sim_matrix_np[class_id]['inter_sim'])}, Std = {np.std(cos_sim_matrix_np[class_id]['inter_sim'])}, Var = {np.var(cos_sim_matrix_np[class_id]['inter_sim'])}\n")

def get_stats_agg(cos_sim_matrix_np):
    
    intra_sim_all,inter_sim_all = [], []

    for class_id in cos_sim_matrix_np:
        intra_sim_all.extend(cos_sim_matrix_np[class_id]['intra_sim'])
        inter_sim_all.extend(cos_sim_matrix_np[class_id]['inter_sim'])

    mean_intra_sim = np.mean(intra_sim_all)
    std_intra_sim = np.std(intra_sim_all)
    var_intra_sim = np.var(intra_sim_all)

    mean_inter_sim = np.mean(inter_sim_all)
    std_inter_sim = np.std(inter_sim_all)
    var_inter_sim = np.var(inter_sim_all)

    print(f"Overall Statistics:\nIntra-class Similarity: Mean = {mean_intra_sim}, Std = {std_intra_sim}, Var = {var_intra_sim}\nInter-class similarity: Mean = {mean_inter_sim}, Std = {std_inter_sim}, Var = {var_inter_sim}\n")

    mean_intra_sim = round(mean_intra_sim, 4)
    std_intra_sim = round(std_intra_sim, 4)
    mean_inter_sim = round(mean_inter_sim, 4)
    std_inter_sim = round(std_inter_sim, 4)
    return mean_intra_sim, std_intra_sim, mean_inter_sim, std_inter_sim

def report_results(model=None,train_set=None,eval_set=None,train_acc='None',test_acc='None',intra_μ=None,intra_σ=None,inter_μ=None,inter_σ=None, adj_intra_μ=None, adj_intra_σ=None, adj_inter_μ=None, adj_inter_σ=None):
    data = {
        'Architecture': [model],
        'Training_Set': [train_set],
        'Evaluation_Set': [eval_set],
        'Train_Accuracy': [train_acc],
        'Test_Accuracy': [test_acc],
        'Class_Avg_Intra-Mean': [intra_μ],
        'Class_Avg_Intra-Std': [intra_σ],
        'Class_Avg_Inter-Mean': [inter_μ],
        'Class_Avg_Inter-Std': [inter_σ],
        'Class_Avg_Adjusted_Intra-Mean': [adj_intra_μ],
        'Class_Avg_Adjusted_Intra-Std': [adj_intra_σ],
        'Class_Avg_Adjusted_Inter-Mean': [adj_inter_μ],
        'Class_Avg_Adjusted_Inter-Std': [adj_inter_σ]
    }

    df = pandas.DataFrame(data)

    df.to_csv(f'final_results.txt', sep='\t', index=False, mode='a',header= (not os.path.exists(f'final_results.txt')))

def get_acc(info_path):
    #TODO
    pandas.read_csv(info_path, sep='\t')