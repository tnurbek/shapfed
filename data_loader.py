import math 
import random 
import numpy as np

import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset 

from config import get_config
config, unparsed = get_config() 


################################ SYNTHETIC DATASET #########################################

def generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices):
    """
    Generate random data for each secondary center associated with a primary class.
    
    Parameters:
    - n_points_per_center: number of data points per secondary center
    - primary_centers: list of primary center points
    - secondary_offsets: list of relative offsets for the secondary centers
    - cov_matrices: list of 2x2 covariance matrices for each secondary center
    
    Returns:
    - data: tensor containing the data points
    - labels: tensor containing the primary class labels
    """
    data_list = []
    label_list = []

    for label, primary_center in enumerate(primary_centers):
        for j, (offset, cov_matrix) in enumerate(zip(secondary_offsets, cov_matrices)):
            # Calculate secondary center
            secondary_center = primary_center + offset

            # Generate random data points and apply the covariance transformation
            raw_data = torch.randn(n_points_per_center, 2)
            if label == 0:
                transformed_data = raw_data @ torch.tensor([[0.05, 0.], [0., 0.45]]) + secondary_center
            elif label == 3:
                transformed_data = raw_data @ torch.tensor([[0.25, 0.], [0., 0.05]]) + secondary_center
            # elif label == 1:
            #     transformed_data = raw_data @ torch.tensor([[0.01, 0.05], [0.05, 0.01]]) + secondary_center
            else:
                transformed_data = raw_data @ cov_matrix.T + secondary_center
            
            data_list.append(transformed_data)

            # Assign primary label to each data point 
            if label in [0, 1]:
                if j % 2 == 0:
                    labels = torch.full((n_points_per_center,), 0, dtype=torch.long)
                else:
                    labels = torch.full((n_points_per_center,), 1, dtype=torch.long)
            else:
                if j % 2 == 0:
                    labels = torch.full((n_points_per_center,), 2, dtype=torch.long)
                else:
                    labels = torch.full((n_points_per_center,), 3, dtype=torch.long)
            label_list.append(labels)

    return torch.cat(data_list, 0), torch.cat(label_list, 0)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # sample = {'data': self.data[idx], 'label': self.labels[idx]}
        sample = [self.data[idx], self.labels[idx]]
        return sample 


def get_custom_train_loader(batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_num=5, split='heterogeneous'):
    # define transforms
    
    n_points_per_center = 1000 
    # Parameters
    primary_centers = [torch.tensor([-4, 4]),  # top left 
                    torch.tensor([4, 4]),   # top right 
                    torch.tensor([-4, -4]), # bottom left 
                    torch.tensor([4, -4])]  # bottom right 


    secondary_offsets = [torch.tensor([0, 0]), 
                        torch.tensor([1, -1]), 
                        torch.tensor([2, -2]), 
                        torch.tensor([3, -3])] 

    cov_matrices = [torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]),] 

    # Generate data
    data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices) 
    
    # 2 to 5D 
    x = data[:, 0].unsqueeze(1)  # unsqueeze adds a new dimension, making it a column vector
    y = data[:, 1].unsqueeze(1) 
    x2 = x**2
    y2 = y**2
    xy = x * y
    
    data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
    labels = torch.Tensor(labels)  # .long()
    
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)

    data = (data - mean_values) / std_values 

    # Create an instance of the CustomDataset using the previously generated data and labels 
    dataset = CustomDataset(data, labels)
    
    # Create a DataLoader
    
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    is_iid, pnumber = False, model_num 
    
    if pnumber == 1:
        return [
                    torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                    )
                ]

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.labels)))
    num_classes = len(set(np.array(dataset.labels)))

    # dictionary of labels map
    labels = np.array(dataset.labels)
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp 

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if split == 'homogeneous': 
            probs.append([1.0 / pnumber] * pnumber)
        elif split == 'imbalanced': 
            rho = config.rho 
            n = config.alloc_n 
            major_allocation = [rho] * n 
            remaining_allocation = [(1 - sum(major_allocation)) / (model_num - n)] * (model_num - n) 
            prob = major_allocation + remaining_allocation 
            probs.append(prob) 
        else:  # heterogeneous 
            if pnumber == 4: 
                if i == 0:
                    probs.append([0.75, 0.25, 0.0, 0.0]) 
                elif i == 1:
                    probs.append([0.15, 0.85, 0.0, 0.0]) 
                elif i == 2:
                    probs.append([0.0, 0.0, 0.05, 0.95]) 
                elif i == 3:
                    probs.append([0.0, 0.0, 0.5, 0.5]) 
            else:
                break 

    print(probs, end="\n\n") 

    # division 
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id 
    
    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]
    
    for pi in range(pnumber): 
        counts = [0] * 4 
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader] 


def get_custom_test_loader(batch_size, random_seed, shuffle=False, num_workers=4, pin_memory=True, model_num=5, split=''): 

    # Parameters 
    n_points_per_center = 400 
    primary_centers = [torch.tensor([-4, 4]),  # top left 
                    torch.tensor([4, 4]),   # top right 
                    torch.tensor([-4, -4]), # bottom left 
                    torch.tensor([4, -4])]  # bottom right 


    secondary_offsets = [torch.tensor([0, 0]), 
                        torch.tensor([1, -1]), 
                        torch.tensor([2, -2]), 
                        torch.tensor([3, -3])] 

    cov_matrices = [torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]),] 

    # Generate data
    data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices)
    
    # 2 to 5D 
    x = data[:, 0].unsqueeze(1)  # unsqueeze adds a new dimension, making it a column vector
    y = data[:, 1].unsqueeze(1) 
    x2 = x**2
    y2 = y**2
    xy = x * y
    
    data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
    labels = torch.Tensor(labels) # .long()
    
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)

    data = (data - mean_values) / std_values 

    # Create an instance of the CustomDataset using the previously generated data and labels
    dataset = CustomDataset(data, labels)
    
    # Create a DataLoader
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    pnumber = model_num

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.labels)))
    num_classes = len(set(np.array(dataset.labels))) 

    # dictionary of labels map
    labels = np.array(dataset.labels)
    dct = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)
    
    print(class_size)
    print(len(dct[3])) 
    print(num_classes)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes): 
        probs.append([1.0 / pnumber] * pnumber)
    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs): 
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            class_size = len(dct[class_id])
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:]  # to_id
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=False) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 4 
        for label in subsets[pi]: 
            counts[label[1]] += 1  
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [data_loader]


def get_cifar10_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_num=5, split="homogeneous"): 
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, .4468], [0.2470, 0.2435, 0.2616])
    ])

    dataset = datasets.CIFAR10(root=data_dir,
                               transform=trans,
                               download=True,
                               train=True)
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    pnumber = model_num 

    lst = []
    class_size = len(dataset) // len(set(dataset.targets))
    num_classes = len(set(dataset.targets)) 

    # dictionary of labels map and initializing variables 
    labels = dataset.targets
    dct = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp 

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if split == 'homogeneous': 
            probs.append([1.0 / pnumber] * pnumber)
        elif split == 'imbalanced':
            rho = config.rho 
            n = config.alloc_n 
            major_allocation = [rho] * n 
            remaining_allocation = [(1 - sum(major_allocation)) / (model_num - n)] * (model_num - n) 
            prob = major_allocation + remaining_allocation 
            probs.append(prob) 
        else:  # heterogeneous 
            if pnumber == 2: 
                if i < 2:
                    probs.append([1.0, 0.0])
                else:
                    probs.append([0.0, 1.0])
            elif pnumber == 4: 
                if i == 0:
                    probs.append([1.0, 0.0, 0.0, 0.0])
                else:
                    probs.append([0.25, 0.25, 0.25, 0.25]) 


    print(probs, end="\n\n") 
    
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id 
    
    print("[data_loader.py: ] Number of common data points:", len(list(set(lst[0]) & set(lst[1])))) 
    
    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 10
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader]


def get_cifar10_test_loader(data_dir, batch_size, random_seed, num_workers=4, pin_memory=True, model_num=5, split='homogeneous'): 
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, 0.4468], [0.2470, 0.2435, 0.2616])
    ])

    # load dataset
    dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=trans
    )
    
    pnumber = model_num 

    lst = []
    class_size = 200 
    num_classes = len(set(dataset.targets)) 

    # dictionary of labels map
    labels = dataset.targets
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        probs.append([1.0 / pnumber] * pnumber)

    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)}
    for class_id, distribution in enumerate(probs):
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=False) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 10
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [data_loader]