import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train(model,train_loader, cost, optimizer, num_epochs,device):
    
    total_step = len(train_loader)
   
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            ## forward pass ##
            outputs = model(images)
            loss = cost(outputs, labels)

            ## backwards pass and optimizer step (learning) ##
            optimizer.zero_grad()  # zeroes out the gradients, removing exisitng ones to avoid accumulation
            loss.backward()  # gradient of loss, how much each parameter contributed to the loss
            optimizer.step()  # adjusts parameters based on results to minimize loss

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
    return model.state_dict()    


def cos_sim(model, dataset_loader, dataset_name, num_epochs, device, execution_id):
    
    model.eval()
    model.load_state_dict(torch.load(f'../data/{execution_id}/model_{dataset_name}_epoch_{num_epochs+1}.pth'))
    model.to(device)

    class_similarities = {}

    with torch.no_grad():
        for inputs, labels in dataset_loader:
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
                    else:                           #between class
                        class_id_i, class_id_j = labels[i].item(), labels[j].item()

                        if class_id_i not in class_similarities:
                            class_similarities[class_id_i] = {'intra_sim': [], 'inter_sim': []}

                        if class_id_j not in class_similarities:
                            class_similarities[class_id_j] = {'intra_sim': [], 'inter_sim': []}
                            
                        

                        class_similarities[class_id_i]['inter_sim'].append(sim.item())
                        class_similarities[class_id_j]['inter_sim'].append(sim.item())

    # with open(f'model_{dataset_name}_epoch{num_epochs+1}_cos_sim', 'wb') as handle:
    #     pickle.dump(class_similarities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return class_similarities
    

def eval(model, test_dataloader, dataset_name, num_epochs, device, execution_id):
    model.eval()
    model.load_state_dict(torch.load(f'../data/{execution_id}/model_{dataset_name}_epoch_{num_epochs+1}.pth'))
    model.to(device)
    with torch.no_grad():
        feature_vectors, labels_list = [], []
        correct, total = 0,0
        for images, labels in test_dataloader:

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

        return feature_vectors
    

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
        axs[0].hist(cos_sim_matrix_np[class_id]['intra_sim'], bins=50, color='blue', edgecolor='black', alpha=0.5,histype='point')
        axs[0].set_title(f"Intra-class similarity")
        axs[0].set_xlabel("cosine similarity")
        axs[0].set_ylabel("Frequency")

        # Inter-class similarity
        axs[1].hist(cos_sim_matrix_np[class_id]['inter_sim'], bins=50, color='red', edgecolor='black', alpha=0.5)
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

