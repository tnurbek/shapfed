"""
- approximation of shapley value; 
- usage of the controlled number of communications 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix

from config import get_config 
from models import SimpleNetwork, Model 
from data_loader import get_custom_train_loader, get_custom_test_loader 

import wandb 
from torch.utils.tensorboard import SummaryWriter 

from itertools import combinations
import math

# initialization 
config, unparsed = get_config() 

num_workers = 4 
pin_memory = True 
model_num = 6
intersection = config.intersection 
split = config.split 
batch_size = config.batch_size 
random_seed = config.random_seed 
use_tensorboard = config.use_tensorboard 
use_wandb = True
aggregation = 2
input_dim = 5 

data_dir = '' 
save_name = 'sgdm0001' 

file_name = f'{save_name}_p{model_num}_batch{batch_size}_inter{intersection}_seed{random_seed}_agg{aggregation}' 

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if using wandb 
if use_wandb:
    wandb.init(project="federated_learning_with_shapley", name=f"{file_name}") 

# if you're using tensorboard 
if use_tensorboard:
    writer = SummaryWriter(log_dir=f'./logs/{file_name}/') 

# data loader 
kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'model_name': save_name, 'model_num': model_num, 'intersection': intersection, 'split': split}
from FLamby.flamby.datasets.fed_isic2019 import *
test_data_loader = [torch.utils.data.DataLoader(FedIsic2019(center = i, train = False, pooled = False), batch_size = batch_size, shuffle = False, num_workers = 4,) for i in range(NUM_CLIENTS)]
server_val_loader = [torch.utils.data.DataLoader(FedIsic2019(train = False, pooled = True), batch_size = batch_size, shuffle = False, num_workers = 4,)]
train_data_loader = [torch.utils.data.DataLoader(FedIsic2019(center = i, train = True, pooled = False), batch_size = batch_size, shuffle = True, num_workers = 4,) for i in range(NUM_CLIENTS)]

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

def compute_shapley_value_new(clients, dataloader, weights):
    n = len(clients)
    v = {}  # Dictionary to store value for each subset 
    class_v = {}

    # Evaluate for each subset that includes client_idx
    for size in range(1, n + 1): 
    # for size in range(n - 1, n + 1):
        subsets = [subset for subset in combinations(range(n), size)] 
        
        for subset in subsets:
            if subset not in v:
                # Create a temporary server for this subset 
                subset_clients = [clients[i] for i in subset] 
                curr_weights = [weights[j] for j in subset] 
                normalized_curr_weights = curr_weights / np.sum(curr_weights)

                temp_server = Server(subset_clients) 
                temp_server.aggregate(coefficients = normalized_curr_weights) 

                # Calculate class-wise accuracy and store it
                acc = classwise_accuracy(temp_server.model, dataloader) 
                v[subset] = np.mean(acc) 
                # v[subset] = np.sum([x for x in acc if x > 0]) / n 
                class_v[subset] = acc 

    # Compute Shapley value for client_idx
    shapley_values, class_shapley_values = [], [] 
    for cl_idx in range(n):
        shapley_value, class_shapley_value = 0, 0 
        # for size in range(1, n + 1):
        for size in range(1, n + 1): 
            for subset in combinations(range(n), size): 
                if cl_idx in subset:
                    # Calculate marginal contribution
                    subset_without_client = tuple([i for i in subset if i != cl_idx]) 
                    marginal_contribution = v[subset] - v.get(subset_without_client, 0)
                    class_marginal_contribution = class_v[subset] - class_v.get(subset_without_client, 0) 

                    # Weight by the number of permutations 
                    weight = math.factorial(size - 1) * math.factorial(n - size) / math.factorial(n) 
                    shapley_value += weight * marginal_contribution 
                    class_shapley_value += weight * class_marginal_contribution 

        shapley_values.append(shapley_value) 
        # shapley_values.append(np.sum([x for x in class_shapley_value if x > 0]) / n) 
        class_shapley_values.append(class_shapley_value) 

    return shapley_values, class_shapley_values 

def compute_approximate_shapley_value(clients, dataloader, weights): 
    n = len(clients) 
    similarity_matrix = torch.zeros((n, 4)) 
    layer_name = 'model.classifier.1.weight' 
    subsets = [subset for subset in combinations(range(n), n)] 
    for subset in subsets: 
        # Create a temporary server for this subset 
        subset_clients = [clients[i] for i in subset] 
        curr_weights = [weights[j] for j in subset] 
        normalized_curr_weights = curr_weights / np.sum(curr_weights)

        temp_server = Server(subset_clients) 
        temp_server.aggregate(coefficients = normalized_curr_weights) 

        # use approximation method to compute CSSV 
        num_classes = temp_server.model.state_dict()[layer_name].shape[0]
        for cls_id in range(num_classes):
            w1 = temp_server.model.state_dict()[layer_name][cls_id].view(1, -1)  # unnormalized 
            # w1 = F.normalize(temp_server.model.state_dict()[layer_name][cls_id].view(1, -1), p=2)  # normalized 
            for client_id in range(len(subset)): 
                # unnormalized 
                w2 = subset_clients[client_id].model.state_dict()[layer_name][cls_id].view(1, -1)
                sim = F.cosine_similarity(w1, w2) 
                similarity_matrix[client_id][cls_id] = sim 

                # normalized 
                # w2 = F.normalize(subset_clients[client_id].model.state_dict()[layer_name][cls_id].view(1, -1), p=2)
                # sim = F.cosine_similarity(w1, w2) 
                # similarity_matrix[client_id][cls_id] = sim 
    
    shapley_values = np.array(similarity_matrix.mean(axis=1))
    return shapley_values, similarity_matrix 


def compute_approximate_shapley_value_3(clients, dataloader, weights): 
    n = len(clients) 
    num_classes = clients[0].model.state_dict()['model.classifier.1.weight'].shape[0]
    similarity_matrix = torch.zeros((n, num_classes))  # One similarity value per class
    
    weight_layer_name = 'model.classifier.1.weight'
    bias_layer_name = 'model.classifier.1.bias'

    subsets = [subset for subset in combinations(range(n), n)] 
    for subset in subsets: 
        # Create a temporary server for this subset 
        subset_clients = [clients[i] for i in subset] 
        curr_weights = [weights[j] for j in subset] 
        normalized_curr_weights = curr_weights / np.sum(curr_weights)

        temp_server = Server(subset_clients) 
        temp_server.aggregate(coefficients=normalized_curr_weights) 

        for cls_id in range(num_classes):
            # Concatenate weight and bias for the class in the aggregated model
            w1 = torch.cat([
                temp_server.model.state_dict()[weight_layer_name][cls_id].view(-1),
                temp_server.model.state_dict()[bias_layer_name][cls_id].view(-1)
            ]).view(1, -1)
            w1 = F.normalize(w1, p=2)

            for client_id in range(len(subset)):
                # Concatenate weight and bias for the class in the client's model
                w2 = torch.cat([
                    subset_clients[client_id].model.state_dict()[weight_layer_name][cls_id].view(-1),
                    subset_clients[client_id].model.state_dict()[bias_layer_name][cls_id].view(-1)
                ]).view(1, -1)
                w2 = F.normalize(w2, p=2)

                # Compute cosine similarity
                sim = F.cosine_similarity(w1, w2).item()
                similarity_matrix[client_id][cls_id] = sim
    
    shapley_values = torch.mean(similarity_matrix, dim=1).numpy()
    return shapley_values, similarity_matrix 


def compute_true_approximate_shapley_value(clients, dataloader, weights): 
    n = len(clients) 
    class_v = {}
    layer_name = 'model.classifier.1.weight' 
    for size in range(1, n+1):
        subsets = [subset for subset in combinations(range(n), size)] 
        for subset in subsets: 
            # Create a temporary server for this subset 
            subset_clients = [clients[i] for i in subset] 
            curr_weights = [weights[j] for j in subset] 
            normalized_curr_weights = curr_weights / np.sum(curr_weights)

            temp_server = Server(subset_clients) 
            temp_server.aggregate(coefficients = normalized_curr_weights) 

            # use approximation method to compute CSSV 
            num_classes = temp_server.model.state_dict()[layer_name].shape[0]  # 4 
            acc = np.zeros(4) 
            for client_id in range(len(subset)): 
                for cls_id in range(num_classes):
                    w1 = temp_server.model.state_dict()[layer_name][cls_id].view(1, -1)  # unnormalized  
                    w2 = subset_clients[client_id].model.state_dict()[layer_name][cls_id].view(1, -1)
                    sim = F.cosine_similarity(w1, w2) 
                    acc[cls_id] += sim 
            acc /= len(subset) 
            class_v[subset] = acc 
    
    class_shapley_values = [] 
    for client_id in range(n):
        class_shapley_value = 0 
        # for size in range(1, n + 1):
        for size in range(1, n + 1): 
            for subset in combinations(range(n), size): 
                if client_id in subset: 
                    # Calculate marginal contribution 
                    subset_without_client = tuple([i for i in subset if i != client_id])
                    class_marginal_contribution = class_v[subset] - class_v.get(subset_without_client, 0) 

                    # Weight by the number of permutations 
                    weight = math.factorial(size - 1) * math.factorial(n - size) / math.factorial(n) 
                    class_shapley_value += weight * class_marginal_contribution 

        # shapley_values.append(np.sum([x for x in class_shapley_value if x > 0]) / n) 
        class_shapley_values.append(class_shapley_value) 

    shapley_values = np.array(class_shapley_values).mean(axis=1) 
    return shapley_values, class_shapley_values  


# Client Class
class Client:
    def __init__(self, dataloader, client_id):
        self.model = Model().cuda()
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
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


# Server Class
class Server:
    def __init__(self, clients):
        self.model = Model().cuda()
        self.clients = clients

    def aggregate(self, coefficients):
        total_weights = None

        for client_id, client in enumerate(self.clients): 
            client_weights = client.get_weights()

            if total_weights is None:
                total_weights = {name: torch.zeros_like(param).float() for name, param in client_weights.items()}

            for name, param in client_weights.items():
                total_weights[name] += torch.tensor(coefficients[client_id]).float() * param.float() 

        # for name in total_weights: 
        #     total_weights[name] /= len(self.clients) 
        prev_weights = self.model.state_dict()
        eta = 0.9 
        for name, param in total_weights.items():
            prev_weights[name] = (1 - eta) * prev_weights[name] + eta * param 

        self.model.load_state_dict(prev_weights) 

    def broadcast(self, coefficients): 
        for client in self.clients: 
            for (global_param, client_param) in zip(self.model.parameters(), client.model.parameters()):
                # I approach 
                # update = global_param.data - client_param.data 
                # personalized_update = update * coefficients[client.client_id] 
                # client_param.data += personalized_update 
                # II approach 
                client_param.data = (1 - coefficients[client.client_id]) * client_param.data + coefficients[client.client_id] * global_param.data 
            # III approach 
            # client_state_dict = client.model.state_dict() 
            # for name, global_param in self.model.named_parameters():
            #     if name in ['fc5.weight','fc5.bias']: 
            #         client_param = client_state_dict[name]
            #         update = global_param.data - client_param
            #         personalized_update = update * coefficients[client.client_id]
            #         client_param += personalized_update
            #     else: 
            #         # For other layers, use standard updates (or apply a different strategy)
            #         client_state_dict[name] = global_param.data 
            # client.model.load_state_dict(client_state_dict) 
        # 0 approach 
        # client.set_weights(self.model.state_dict())
    
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

clients = [Client(loader, i) for i, loader in enumerate(client_loaders)]
server = Server(clients) 
weights = [1 / model_num] * model_num 
static_shapley_values = None 
shapley_values, mu = None, 0.95 
freq_rounds = None 

# Federated Averaging (FedAvg) Algorithm 
# num_rounds = config.num_rounds 
num_rounds = 50
num_lepochs = [config.num_lepochs] * model_num 

for round in range(num_rounds):

    # if freq_rounds is not None:
    #     import copy
    #     server_model = copy.deepcopy(server.model.state_dict())
    #     server = Server(clients=[x for x in clients if round % freq_rounds[x.client_id] == 0])
    #     server.model.load_state_dict(server_model)
    
    for cl_idx, client in enumerate(clients): 
        client.train(epochs = num_lepochs[cl_idx]) 
    
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

    
    # here, I just used server.evaluate, since the test set is balanced;
    # ideally, in fed-isic case, change it class-wise accuracy (balanced): Toluwani.
    val_accuracy = classwise_accuracy(server.model, server_val_loader[-1])
    print(f"Round {round + 1}/{num_rounds}, Server Accuracy: {np.mean(val_accuracy)*100:.2f}%, {val_accuracy}")

    # wandb 
    if use_wandb:
        wandb.log({f"valid_acc": np.mean(val_accuracy) * 100, "round": round}) 
    
    # tensorboard 
    if use_tensorboard:
        writer.add_scalar(f"valid_acc", np.mean(val_accuracy) * 100, global_step=round)
    
    # Compute Shapley values for each client 
    temp_shapley_values, temp_class_shapley_values = compute_approximate_shapley_value_3(clients, test_data_loader[0], weights) 
    if shapley_values is None:
        shapley_values = np.array(temp_shapley_values)
        class_shapley_values = np.array(temp_class_shapley_values)
    else:
        shapley_values = mu * shapley_values + (1 - mu) * temp_shapley_values 
        class_shapley_values = mu * class_shapley_values + (1 - mu) * np.array(temp_class_shapley_values) 

    normalized_shapley_values = shapley_values / np.sum(shapley_values) 

    print(shapley_values, normalized_shapley_values) 
    print(class_shapley_values) 

    # arr = np.array(class_shapley_values) 
    # class_shapley_values = (arr - arr.min(axis=0)) / arr.max(axis=0) 
    # print(class_shapley_values) 

    # update static_shapley_values only at the start 
    if static_shapley_values is None: 
        static_shapley_values = normalized_shapley_values
        freq_rounds = [math.floor(x) for x in np.array(static_shapley_values) / min(np.array(static_shapley_values))] 
    
    # update the number of local epochs 
    # adaptive local rounds: [for heterogeneous settings, try using just 1]
    if len(clients) == model_num: # and split != 'heterogeneous': 
        for idx, cls_sv in enumerate(class_shapley_values): 
            num_lepochs[idx] = len([x for x in cls_sv if x >= cls_sv.mean()]) 
            # you can change this condition x >= cls_sv.mean() to be some kind of hypothesis test, 
            # say you use std to be close to 0 to have the full length 

    for i, shap_value in enumerate(shapley_values):
        # Logging to wandb
        if use_wandb:
            wandb.log({f"shapley_value_{i}": shap_value, "round": round})
        
        # Logging to TensorBoard 
        if use_tensorboard:
            writer.add_scalar(f"client_{i}/shapley_value", shap_value, global_step=round) 
    
    print('#' * 100, end="\n\n") 

    if use_wandb:
        wandb.log({f"class_shapley_value_{round}": wandb.Table(dataframe=pd.DataFrame(class_shapley_values)), "round": round})


    # aggregate the weights and broadcast 
    if aggregation == 1: 
        server.aggregate(coefficients = weights) 
    else:
        weights = normalized_shapley_values  # normalized_shapley_values 
        server.aggregate(coefficients = weights) 
    server.broadcast(coefficients = shapley_values) 