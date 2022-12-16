import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SimpleModel(nn.Module):
    # Simple Model for Normalized LMS Training on MNIST
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding=1,bias=False)
        #self.bath1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,padding=1, stride=2, bias=False)
        #self.bath2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,padding=1, stride=2, bias=False)
        #self.bath3 = nn.BatchNorm2d(32)
        #self.drop1 = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(32*7*7, 10)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        #x = self.bath1(x)
        x = self.conv2(x)
        #x = self.bath2(x)
        x = F.relu(x)
        x = self.conv3(x) 
        #x = self.bath3(x)
        #x = self.drop1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        return x 
    
    def inputs_ext(self, x):
        
        out1 = self.conv1(x)
        #out2 = self.bath1(out1)
        out3 = self.conv2(out1)
        #out4 = self.bath2(out3)
        out5 = F.relu(out3)
        out6 = self.conv3(out5) 
        #out7 = self.bath3(out6)
        #out8 = self.drop1(out7)
        out9 = F.relu(out6)
        out10 = torch.flatten(out9, 1)
        
        return [x, out1, out5, out10]

class SimpleModel_Cifar(nn.Module):
    # Simple Model for Normalized LMS Training
    def __init__(self):
        super(SimpleModel_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,padding=1,bias=False)
        self.bath1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,padding=1, stride=2, bias=False)
        self.bath2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,padding=1, stride=2, bias=False)
        self.bath3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,3,padding=1, stride=2, bias=False)
        self.bath4 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(64*4*4, 10)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bath1(x)
        x = self.conv2(x)
        x = self.bath2(x)
        x = F.relu(x)
        x = self.conv3(x) 
        x = self.bath3(x)
        x = F.relu(x)
        x = self.conv4(x) 
        x = self.bath4(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        return x 
    
    def inputs_ext(self, x):
        out1 = self.conv1(x)
        out2 = self.bath1(out1)
        out3 = self.conv2(out2)
        out4 = self.bath2(out3)
        out5 = F.relu(out4)
        out6 = self.conv3(out5) 
        out7 = self.bath3(out6)
        out8 = F.relu(out7)
        out9 = self.conv4(out8) 
        out10 = self.bath4(out9)
        out11 = self.drop1(out10)
        out12 = F.relu(out11)
        out13 = torch.flatten(out12, 1)
        
        return [x, out2, out5, out8, out13]
    
