import torch 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

from config import get_config 
from models import SimpleNetwork 
from data_loader import get_custom_train_loader, get_custom_test_loader 


# initialization 
config, unparsed = get_config() 

num_workers = 4 
pin_memory = True 
model_num = config.model_num 
intersection = config.intersection 

print(intersection) 

batch_size = config.batch_size 
random_seed = config.random_seed 
input_dim = 5 

data_dir = '' 
save_name = config.save_name 

file_name = f'{save_name}_p{model_num}_batch{batch_size}_inter{intersection}_seed{random_seed}' 


# data loader 
kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'model_name': save_name, 'model_num': model_num, 'intersection': intersection}
test_data_loader = get_custom_test_loader(data_dir, batch_size, random_seed, **kwargs)
train_data_loader = get_custom_train_loader(data_dir, batch_size, random_seed, shuffle=True, **kwargs) 


# ------------------------------------------ training methods ------------------------------------------ 
def train_single_epoch(model, train_loader, optimizer):
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 

def validate(model, val_loader):
    model.eval() 
    correct, total = 0, 0 
    with torch.no_grad(): 
        for batch in val_loader: 
            inputs, labels = batch 
            inputs, labels = inputs.cuda(), labels.cuda() 
            outputs = model(inputs) 
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
    return 100 * correct / total 


def validate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset) * 100
    return avg_loss, accuracy


def average_models(global_model, client_models):
    eta = 1.0 
    global_dict = global_model.state_dict() 
    for k in global_dict.keys(): 
        global_dict[k] = (1-eta) * global_dict[k] + eta * torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


# training 
global_model = SimpleNetwork().cuda() 
client_models = [SimpleNetwork().cuda() for _ in range(model_num)] 

criterion = nn.CrossEntropyLoss().cuda()
client_optimizers = [optim.SGD(client_models[i].parameters(), lr=0.01) for i in range(model_num)]

for i in range(model_num):
    client_models[i].load_state_dict(global_model.state_dict()) 

# accuracies = [[]] * (model_num + 1) 
# rounds = 20 
# for round in range(rounds): 
#     # Local training 
#     for i in range(model_num): 
#         train_single_epoch(client_models[i], train_data_loader[i], client_optimizers[i]) 
    
#     # Update global model weights by averaging client model weights
#     global_model = average_models(global_model, client_models)
    
#     # Validation 
#     for i in range(model_num):
#         accuracy = validate(client_models[i], test_data_loader[i]) 
#         accuracies[i].append(accuracy)
    
#     accuracy = validate(global_model, test_data_loader[-1]) 
#     accuracies[-1].append(accuracy) 
#     print(f"Round {round + 1}: P1: {accuracies[0][-1]:.2f}%, P2: {accuracies[1][-1]:.2f}%, GL: {accuracy:.2f}%")
#     print("-" * 75)
    
#     # Broadcast global model to all clients
#     for client in range(model_num): 
#         client_models[client].load_state_dict(global_model.state_dict())


accuracies = [[] for _ in range(model_num + 1)]
losses = [[] for _ in range(model_num + 1)]
rounds = 30

for round in range(rounds):
    # Local training 
    for i in range(model_num): 
        train_single_epoch(client_models[i], train_data_loader[i], client_optimizers[i])
    
    # Update global model weights by averaging client model weights
    global_model = average_models(global_model, client_models)
    
    # Validation 
    for i in range(model_num):
        loss, accuracy = validate_model(client_models[i], test_data_loader[i]) 
        accuracies[i].append(accuracy) 
        losses[i].append(loss) 
    
    loss_gl, accuracy_gl = validate_model(global_model, test_data_loader[-1]) 
    accuracies[-1].append(accuracy_gl)
    losses[-1].append(loss_gl)
    print(f"Round {round + 1}: P1: {accuracies[0][-1]:.2f}%; {losses[0][-1]:.4f}, P2: {accuracies[1][-1]:.2f}%; {losses[1][-1]:.4f}, GL: {accuracy_gl:.2f}%; {loss_gl:.4f}") 
    print("-" * 75)
    
    # Broadcast global model to all clients
    for client in range(model_num): 
        client_models[client].load_state_dict(global_model.state_dict())


pic_name = f'images/img_{file_name}.jpg' 
plt.figure(figsize=(10, 3)) 
plt.plot(accuracies[0], label="P1", color='blue') 
plt.plot(accuracies[1], label="P2", color='red') 
plt.plot(accuracies[2], label="GL", color='black') 
plt.title('Accuracy of parties') 
plt.legend()

plt.savefig(pic_name) 