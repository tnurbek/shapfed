import math 
import random 
import numpy as np

import torch 
from torch.utils.data import Dataset 


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


def get_custom_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0):
    # define transforms
    
    n_points_per_center = 2000
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
    batch_size = 32     
    
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    is_iid, pnumber = False, model_num 
    if "_iid_" in model_name:
        is_iid = True
    
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
    
    until_index = (1 - intersection) * num_classes 

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if is_iid: 
            probs.append([1.0 / pnumber] * pnumber)
        else:
            if i < until_index:
                # if i % 2 == 0:
                if i < 2:
                    probs.append([1.0, 0.0])
                else:
                    probs.append([0.0, 1.0])
            else:
                probs.append([1.0 / pnumber] * pnumber) 
    print(probs, end="\n\n") 

    # division 
    if not is_iid:
        intersection = 0.0 
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            to_id = math.ceil((1 - intersection) * to_id) 
            from_id = to_id 
    
    print("[data_loader.py: ] Number of common data points:", len(list(set(lst[0]) & set(lst[1])))) 
    
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
    
    # for batch in train_loader:
    #     print(batch['data'], batch['label'])
    #     break

    return t_loaders + [train_loader] 






def get_custom_test_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0): 

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
    batch_size = 32     
    
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
