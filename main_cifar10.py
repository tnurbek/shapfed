import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix

from config import get_config 
from models import resnet34_model
from data_loader import get_cifar10_train_loader, get_cifar10_test_loader 


import wandb 
from torch.utils.tensorboard import SummaryWriter 

from itertools import combinations
from tqdm import tqdm
import math
import random 

# initialization 
config, unparsed = get_config() 

num_workers = 4 
pin_memory = True 
model_num = config.model_num 
split = config.split 
batch_size = config.batch_size 
random_seed = config.random_seed 
use_tensorboard = config.use_tensorboard 
use_wandb = config.use_wandb 
aggregation = config.aggregation

data_dir = 'data/cifar10/'
file_name = f'cifar10_p{model_num}_batch{batch_size}_{split}_seed{random_seed}_agg{aggregation}' 

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if using wandb 
if use_wandb:
    wandb.init(project="shapfed", name=f"{file_name}") 

# if using tensorboard 
if use_tensorboard:
    writer = SummaryWriter(log_dir=f'./logs/{file_name}/') 
    
# data loader 
kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'model_num': model_num, 'split': split} 
test_data_loader = get_cifar10_test_loader(data_dir, batch_size, random_seed, **kwargs)
train_data_loader = get_cifar10_train_loader(data_dir, batch_size, random_seed, shuffle=True, **kwargs) 


def classwise_accuracy(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.cuda(), targets.cuda()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return class_acc


def compute_cssv_cifar(clients, weights): 
    n = len(clients) 
    num_classes = 10 # clients[0].model.state_dict()['linear.weight'].shape[0]
    similarity_matrix = torch.zeros((n, num_classes))  # One similarity value per class
    
    weight_layer_name = 'linear.weight'
    bias_layer_name = 'linear.bias'
    subsets = [subset for subset in combinations(range(n), n)] 
    for subset in subsets: 
        # Create a temporary server for this subset 
        subset_clients = [clients[i] for i in subset] 
        curr_weights = [weights[j] for j in subset] 
        # normalized_curr_weights = softmax(curr_weights)  # curr_weights / np.sum(curr_weights)
        normalized_curr_weights = curr_weights / np.sum(curr_weights)


        temp_server = Server(subset_clients) 
        temp_server.aggregate(coefficients=normalized_curr_weights) 
        temp_server.aggregate_gradients(coefficients=normalized_curr_weights) 

        for cls_id in range(num_classes):
            # Use gradients instead of weights and biases
            w1_grad = torch.cat([
                temp_server.stored_grads[weight_layer_name][cls_id].view(-1), 
                temp_server.stored_grads[bias_layer_name][cls_id].view(-1) 
            ]).view(1, -1) 
            w1_grad = F.normalize(w1_grad, p=2) 

            for client_id in range(len(subset)):
                w2_grad = torch.cat([
                    subset_clients[client_id].get_gradients()[weight_layer_name][cls_id].view(-1),
                    subset_clients[client_id].get_gradients()[bias_layer_name][cls_id].view(-1)
                ]).view(1, -1)
                w2_grad = F.normalize(w2_grad, p=2)

                # Compute cosine similarity with gradients
                sim = F.cosine_similarity(w1_grad, w2_grad).item()
                similarity_matrix[client_id][cls_id] = sim 
    
    shapley_values = torch.mean(similarity_matrix, dim=1).numpy()
    return shapley_values, similarity_matrix 


class Client:
    def __init__(self, dataloader, client_id):
        self.model = resnet34_model().cuda()
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01) 
        self.client_id = client_id

    def train(self, epochs=1): 
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs): 
            for data, target in self.dataloader:
                data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
    
    def get_gradients(self): 
        # Collecting and returning the gradients
        gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
        return gradients
    

# Server Class
class Server:
    def __init__(self, clients):
        self.model = resnet34_model().cuda()
        self.clients = clients

        self.stored_grads = None 

    def aggregate_gradients(self, coefficients):
        total_grads = None

        for client_id, client in enumerate(self.clients): 
            client_grads = client.get_gradients() 

            if total_grads is None:
                total_grads = {name: torch.zeros_like(grad) for name, grad in client_grads.items()}

            for name, grad in client_grads.items():
                total_grads[name] += coefficients[client_id] * grad

        # Store the aggregated gradients
        self.stored_grads = total_grads

    def aggregate(self, coefficients):
        total_weights = None

        for client_id, client in enumerate(self.clients): 
            client_weights = client.get_weights()

            if total_weights is None:
                total_weights = {name: torch.zeros_like(param) for name, param in client_weights.items()}

            for name, param in client_weights.items():
                if  total_weights[name].dtype == torch.float32:
                    total_weights[name] += coefficients[client_id] * param.to(torch.float32)
                elif total_weights[name].dtype == torch.int64:
                    total_weights[name] += int(coefficients[client_id]) * param.to(torch.int64)

        prev_weights = self.model.state_dict()
        eta = 1.0 
        for name, param in total_weights.items():
            prev_weights[name] = (1 - eta) * prev_weights[name] + eta * param 

        self.model.load_state_dict(prev_weights) 

    def broadcast(self, coefficients): 
        for client in self.clients: 
            for (global_param, client_param) in zip(self.model.parameters(), client.model.parameters()):
                # personalization 
                client_param.data = (1 - coefficients[client.client_id]) * client_param.data + coefficients[client.client_id] * global_param.data 
    
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss += criterion(output, target).item()

        accuracy = 100 * correct / total
        avg_loss = loss / len(dataloader)
        return accuracy, avg_loss


client_loaders = train_data_loader 

clients = [Client(loader, i) for i, loader in enumerate(client_loaders[:-1])]
server = Server(clients) 
for client in clients:
    client.set_weights(server.model.state_dict()) 
    
weights = [1 / model_num] * model_num 
shapley_values, mu = None, 0.5 
freq_rounds = None 

num_rounds = config.num_rounds 
num_lepochs = [config.num_lepochs] * model_num 

for round in range(num_rounds):
    if round > 0:
        num_lepochs = [1] * model_num 
    
    for cl_idx in tqdm(range(len(clients)), desc=f"Training Clients in Round {round + 1}"):
        client = clients[cl_idx]
        client.train(epochs=num_lepochs[cl_idx])
    
    print('#' * 100)
    
    # evaluating clients 
    for idx, client in enumerate(clients):
        client_val_accuracy = classwise_accuracy(client.model, test_data_loader[idx])  # Use the client's model for evaluation
        print(f"[Client {idx}] Round {round + 1}/{num_rounds}, Balanced Accuracy: {np.mean(client_val_accuracy)*100:.2f}%, {client_val_accuracy}")

        # wandb 
        if use_wandb:
            wandb.log({f"balanced_valid_acc_{idx}": np.mean(client_val_accuracy) * 100, "round": round}) 
            for j, acc in enumerate(client_val_accuracy):
                wandb.log({f"client_{idx}/class_acc_{j}": acc, "round": round}) 
        
        # tensorboard 
        if use_tensorboard:
            writer.add_scalar(f"client_{idx}/balanced_acc", np.mean(client_val_accuracy) * 100, global_step=round) 
            for j, acc in enumerate(client_val_accuracy): 
                writer.add_scalar(f"client_{idx}/class_acc_{j}", acc, global_step=round) 
    

    # here, we just used server.evaluate, since the test set is balanced; 
    val_accuracy, val_loss = server.evaluate(test_data_loader[-1]) 
    print(f"Round {round + 1}/{num_rounds}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}") 
    
    # wandb 
    if use_wandb:
        wandb.log({f"valid_acc": val_accuracy, "round": round}) 
        wandb.log({f"valid_loss": val_loss, "round": round}) 
    
    # tensorboard 
    if use_tensorboard:
        writer.add_scalar(f"valid_acc", val_accuracy, global_step=round) 
        writer.add_scalar(f"valid_loss", val_loss, global_step=round) 
    
    if round % 1 == 0:  # every round 
        # Compute Shapley values for each client every 5 rounds [to be efficient]
        temp_shapley_values, temp_class_shapley_values = compute_cssv_cifar(clients, weights) 
        if shapley_values is None: 
            shapley_values = np.array(temp_shapley_values)
            class_shapley_values = np.array(temp_class_shapley_values)
        else:
            shapley_values = mu * shapley_values + (1 - mu) * temp_shapley_values 
            class_shapley_values = mu * class_shapley_values + (1 - mu) * np.array(temp_class_shapley_values) 

        # normalized_shapley_values = softmax(shapley_values)  # shapley_values / np.sum(shapley_values) 
        normalized_shapley_values = shapley_values / np.sum(shapley_values) 
        broadcast_normalized_shapley_values = shapley_values / np.max(shapley_values)


    print(shapley_values, normalized_shapley_values) 
    print(class_shapley_values) 
    
    print('#' * 100, end="\n\n") 
    
    if aggregation == 1: 
        server.aggregate(coefficients = weights)
    else:  # 2 [shapfed]
        weights = normalized_shapley_values
        server.aggregate(coefficients = weights)
        
        # if one wants to allow the highest-contribution participant to receive full update 
        # server.broadcast(coefficients = broadcast_normalized_shapley_values)

    server.broadcast(coefficients = weights)