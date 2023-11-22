import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from sklearn.metrics import confusion_matrix

from config import get_config 
from models import SimpleNetwork 
from data_loader import get_custom_train_loader, get_custom_test_loader 

import wandb 
from torch.utils.tensorboard import SummaryWriter 

# initialization 
config, unparsed = get_config() 

num_workers = 4 
pin_memory = True 
model_num = config.model_num 
intersection = config.intersection 
batch_size = config.batch_size 
random_seed = config.random_seed 
use_tensorboard = config.use_tensorboard 
use_wandb = config.use_wandb 
input_dim = 5 

data_dir = '' 
save_name = config.save_name 

file_name = f'{save_name}_p{model_num}_batch{batch_size}_inter{intersection}_seed{random_seed}' 

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if using wandb 
if use_wandb:
    wandb.init(project="federated_learning_with_shapley") 

# if you're using tensorboard 
if use_tensorboard:
    writer = SummaryWriter(log_dir=f'./logs/{file_name}/') 

# data loader 
kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'model_name': save_name, 'model_num': model_num, 'intersection': intersection}
test_data_loader = get_custom_test_loader(data_dir, batch_size, random_seed, **kwargs)
train_data_loader = get_custom_train_loader(data_dir, batch_size, random_seed, shuffle=True, **kwargs) 


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


from itertools import combinations
import math

def compute_shapley_value(client_idx, clients, dataloader):
    n = len(clients)
    v = np.zeros(n + 1)  # Store values for each coalition size

    # Evaluate for each coalition size
    for size in range(1, n + 1): 
        subsets = list(combinations(range(n), size))
        
        # For each subset, if the client_idx is in the subset, train and evaluate
        for subset in subsets:
            if client_idx not in subset:
                continue
            
            # Create a temporary server for this subset 
            subset_clients = [clients[i] for i in subset] 
            temp_server = Server(subset_clients) 
            
            # Using one round of FedAvg for simplicity, but more rounds can be used 
            for client in subset_clients: 
                client.train() 
            temp_server.aggregate() 

            # Calculate class-wise accuracy
            acc = classwise_accuracy(temp_server.model, dataloader)
            v[size] += np.mean(acc) 
        
        v[size] /= len(subsets) 

    # Compute Shapley value for client_idx 
    shapley_value = 0 
    for size in range(1, n + 1): 
        weight = math.comb(n - 1, size - 1) / math.comb(n, size) 
        shapley_value += weight * (v[size] - v[size - 1]) 

    return shapley_value 


# Client Class
class Client:
    def __init__(self, dataloader, client_id):
        self.model = SimpleNetwork().cuda()
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.client_id = client_id

    def train(self, epochs=1):
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
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


# Server Class
class Server:
    def __init__(self, clients):
        self.model = SimpleNetwork().cuda()
        self.clients = clients

    def aggregate(self):
        total_weights = None

        for client in self.clients:
            client_weights = client.get_weights()

            if total_weights is None:
                total_weights = {name: torch.zeros_like(param) for name, param in client_weights.items()}

            for name, param in client_weights.items():
                total_weights[name] += param

        for name in total_weights:
            total_weights[name] /= len(self.clients)

        self.model.load_state_dict(total_weights)

    def broadcast(self):
        for client in self.clients:
            client.set_weights(self.model.state_dict())
    
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
    

num_clients = 2
client_loaders = train_data_loader 

clients = [Client(loader, i) for i, loader in enumerate(client_loaders[:-1])]
server = Server(clients)

# Federated Averaging (FedAvg) Algorithm
num_rounds = 50 

for round in range(num_rounds):
    for client in clients:
        client.train()
    
    print('#' * 100)
    # Evaluate the clients
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

    server.aggregate()
    server.broadcast() 
    
    val_accuracy, val_loss = server.evaluate(test_data_loader[-1]) 
    print(f"Round {round + 1}/{num_rounds}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}") 

    # wandb 
    if use_wandb:
        wandb.log({f"valid_acc": val_accuracy, "round": round}) 
    
    # tensorboard 
    if use_tensorboard:
        writer.add_scalar(f"valid_acc", val_accuracy, global_step=round) 
    
    # Compute Shapley values for each client 
    shapley_values = [compute_shapley_value(i, clients, test_data_loader[-1]) for i in range(len(clients))] 
    normalized_shapley_values = shapley_values / np.sum(shapley_values)
    print(shapley_values, normalized_shapley_values) 

    for i, shap_value in enumerate(shapley_values):
        # Logging to wandb
        if use_wandb:
            wandb.log({f"shapley_value_{i}": shap_value, "round": round})
        
        # Logging to TensorBoard 
        if use_tensorboard:
            writer.add_scalar(f"client_{i}/shapley_value", shap_value, global_step=round)     
    
    print('#' * 100, end="\n\n") 