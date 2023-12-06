import torch 
import torch.nn as nn
import torch.nn.functional as F


input_dim = 5 
class SimpleNetwork(nn.Module): 
    def __init__(self): 
        super(SimpleNetwork, self).__init__() 
        self.fc1 = nn.Linear(input_dim, 32) 
        self.fc2 = nn.Linear(32, 64) 
        self.fc3 = nn.Linear(64, 128) 
        self.fc4 = nn.Linear(128, 32) 
        self.fc5 = nn.Linear(32, 4) 
        self.relu = nn.ReLU() 
    
    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x)) 
        x = self.relu(self.fc4(x)) 
        x = self.fc5(x) 
        return x


class Model(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes, pretrained=True):
        super(Model, self).__init__()
        '''
        # Load the pretrained model from the Model Hub. 
        # Note that this was only made for VGG and ResNet models. 
        # You can add your pytorch models here too or create a whole model definition.
        '''
        
        self.model = getattr(models, model_name)(pretrained=pretrained)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.fc(F.relu(self.model(x)))
        return x