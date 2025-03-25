import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
# from torchmetrics import Accuracy


class EEGNet(nn.Module): 
    def __init__(self,num_input,channel,F1,D,fs,num_class,signal_length):
        super().__init__()
        self.signal_length = signal_length
        self.num_input = num_input
        self.F1 = F1
        self.D = D
        self.F2 = D*F1
        self.kernel_size_1= (1,round(fs/2)) 
        self.kernel_size_2= (channel, 1)
        self.kernel_size_3= (1, round(fs/8))
        self.kernel_size_4= (1, 1)
        self.num_class = num_class
        ks0= int(round((self.kernel_size_1[0]-1)/2))
        ks1= int(round((self.kernel_size_1[1]-1)/2))
        self.kernel_padding_1= (ks0, ks1-1)
        ks0= int(round((self.kernel_size_3[0]-1)/2))
        ks1= int(round((self.kernel_size_3[1]-1)/2))
        self.kernel_padding_3= (ks0, ks1)



        # layer 1
        self.conv2d = nn.Conv2d(self.num_input, self.F1, self.kernel_size_1, padding=self.kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(self.F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(self.F1, self.D*self.F1, self.kernel_size_2, groups= self.F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(self.D*self.F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d((1,4))
        self.Dropout = nn.Dropout2d(0.2)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(self.D*self.F1, self.D*self.F1, self.kernel_size_3,
                                                padding=self.kernel_padding_3, groups= self.D*self.F1)
        self.Separable_conv2D_point = nn.Conv2d(self.D*self.F1, self.F2, self.kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(self.F2)
        self.Average_pooling2D_2 = nn.AvgPool2d((1,8))
        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(self.F2*round(self.signal_length/32), self.num_class)
        self.Softmax = nn.Softmax(dim= 1)
        
        
    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x)) #.relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)
        
        return y
    
# Example usage:
# if __name__ == '__main__':
#     fs= 200                  #sampling frequency
#     channel= 22              #number of electrode
#     num_input= 1             #number of channel picture (for EEG signal is always : 1)
#     num_class= 5             #number of classes 
#     signal_length = 200      #number of sample in each tarial

#     F1= 8                    #number of temporal filters
#     D= 3                     #depth multiplier (number of spatial filters)
#     F2= D*F1                 #number of pointwise filters

    
#     model = EEGNet(num_input,F1,D,fs,num_class,signal_length)
    
    
#     # print(model)
#     # Create a dummy input tensor with shape (batch_size, channels, height, width)
#     dummy_input = torch.randn(1,1,22,200)
#     output = model(dummy_input)
#     # print("Output shape:", output.shape)