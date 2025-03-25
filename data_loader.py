import os
import numpy as np
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import normalize_epochs


# 1. load data 
data_path = '/egr/research-slim/liangqi1/EEG_project/EEG-classifier/EEGNet_wcnn/data_sample'
name_lst = os.listdir(data_path)

# assume name_lst[0] is Closed Eyes，name_lst[1] is Silence Meditation data
data_CE = np.load(os.path.join(data_path, name_lst[0]))  # Closed Eyes data，shape (n_epochs, n_channels, n_times)
data_SL = np.load(os.path.join(data_path, name_lst[1]))  # Silence Meditation data

print("Closed Eyes shape:", data_CE.shape)
print("Silence Meditation shape:", data_SL.shape)


data_CE = normalize_epochs(data_CE)
data_SL = normalize_epochs(data_SL)

# 2. add one dim (n_epochs, 1, n_channels, n_times)）
data_CE = np.expand_dims(data_CE, axis=1)
data_SL = np.expand_dims(data_SL, axis=1)

# 4. generate the label
# assume Closed Eyes label is 0，Silence Meditation label is 1
labels_CE = np.zeros(data_CE.shape[0])
labels_SL = np.ones(data_SL.shape[0])

# concat the data
data_all = np.concatenate((data_CE, data_SL), axis=0)
labels_all = np.concatenate((labels_CE, labels_SL), axis=0)

# shuffling
data_all, labels_all = shuffle(data_all, labels_all, random_state=42)

print("Combined data shape:", data_all.shape)
print("Combined labels shape:", labels_all.shape)

# 5. train and test separation（e.g. 80% training，20% testing）
data_tensor = torch.tensor(data_all, dtype=torch.float32)
labels_tensor = torch.tensor(labels_all, dtype=torch.long)

train_data, test_data, train_labels, test_labels = train_test_split(
    data_tensor, labels_tensor, test_size=0.2, random_state=42
)

# 6. create PyTorch DataLoader
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

batch_size = 4  # can change to any

g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


