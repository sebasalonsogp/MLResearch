import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def train(model=None,train_loader=None, cost=None, optimizer=None, num_epochs=None, device=None):

    if model is None:
        raise ValueError("Model is not defined")
    elif train_loader is None:
        raise ValueError("Dataset is not defined")
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

    total_step = len(train_loader)
   
    for epoch in range(num_epochs):

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

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
    return model.state_dict()    


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
    

def eval(model=None, eval_dataloader=None, device=None):
        
    if model is None:
        raise ValueError("Model is not defined")
    elif eval_dataloader is None:
        raise ValueError("Dataset is not defined")
    elif device is None:
        raise ValueError("Device is not defined")


    print(f"Evaluating the model...")

    model.eval()
    model.to(device)
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
    

def plot_line_graph(cos_sim_matrix_np,sample_size=1000):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle("Cosine Similarity Line Graph")

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


def plot_hist(cos_sim_matrix_np):

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

def aggregated_hist(cos_sim_matrix_np,histtype='step'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Aggregated Cosine Similarity Histogram")

    # Intra-class similarity
    #all_intra_sim = np.array([sim for class_id in cos_sim_matrix_np for sim in cos_sim_matrix_np[class_id]['intra_sim']])
    colors_intra = cm.get_cmap('Blues', len(cos_sim_matrix_np))
    for i, class_id in enumerate(cos_sim_matrix_np):
        axs[0].hist(cos_sim_matrix_np[class_id]['intra_sim'], bins=50, color=colors_intra(i), edgecolor='black', alpha=0.5, histtype=histtype, label=f"Class {class_id}")
    axs[0].set_title(f"Intra-class similarity")
    axs[0].set_xlabel("cosine similarity")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # Inter-class similarity
    #all_inter_sim = np.array([sim for class_id in cos_sim_matrix_np for sim in cos_sim_matrix_np[class_id]['inter_sim']])
    colors_inter = cm.get_cmap('Reds', len(cos_sim_matrix_np))
    for j, class_id in enumerate(cos_sim_matrix_np):
        axs[1].hist(cos_sim_matrix_np[class_id]['inter_sim'], bins=50, color=colors_inter(j), edgecolor='black', alpha=0.5, histtype='step', label=f"Class {class_id}")
    axs[1].set_title(f"Inter-class similarity")
    axs[1].set_xlabel("cosine similarity")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    # Show the figure
    plt.show()
    

def print_res(cos_sim_matrix_np):
    for class_id in cos_sim_matrix_np:
        print(f"Class {class_id}:\nIntra-class Similarity: Mean = {np.mean(cos_sim_matrix_np[class_id]['intra_sim'])}, Std = {np.std(cos_sim_matrix_np[class_id]['intra_sim'])}, Var = {np.var(cos_sim_matrix_np[class_id]['intra_sim'])}\nInter-class similarity: Mean = {np.mean(cos_sim_matrix_np[class_id]['inter_sim'])}, Std = {np.std(cos_sim_matrix_np[class_id]['inter_sim'])}, Var = {np.var(cos_sim_matrix_np[class_id]['inter_sim'])}\n")
