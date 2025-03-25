import numpy as np
import os
from models import EEGNet
import torch
import torch.nn as nn
import torch.optim as optim
import data_loader
from tqdm import trange
from torchmetrics import Accuracy
from torcheval.metrics.functional import multiclass_f1_score
import matplotlib.pyplot as plt
import time

timestamp = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = f'/egr/research-slim/liangqi1/EEG_project/EEG-classifier/EEGnet/checkpoints/model_{timestamp}.pth'
figure_path = f'/egr/research-slim/liangqi1/EEG_project/EEG-classifier/EEGnet/figures/training_curves_{timestamp}.png'


torch.cuda.empty_cache()
gpu_no = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)
use_gpu = True
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_epochs = 101
lr = 1e-4
# Parameters
fs = 128                  # sampling frequency
channel = 65              # number of electrodes
num_input = 1             # number of channel pictures (for EEG signal is always: 1)
num_class = 2             # number of classes 
signal_length = 192       # number of samples in each trial (1.5s)

F1 = 16                    # number of temporal filters
D = 2                     # depth multiplier (number of spatial filters)

# load model
model = EEGNet(num_input, channel, F1, D, fs, num_class, signal_length).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.NAdam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


train_loss = []
val_loss = []
f1_scores = []  # List to store the validation F1 score for each epoch

best_f1 = 0.0
for epoch in trange(num_epochs):
    train_loss_total = 0
    num_batches = 0

    # Training loop
    for i, (inputs, targets) in enumerate(data_loader.train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        train_loss_total += loss.item()
        num_batches += 1 
    
    # Validation loop: accumulate predictions and targets for F1 calculation
    all_preds = []
    all_targets = []
    val_loss_total = 0
    num_batches_val = 0

    with torch.no_grad():
        for i, (val_inputs, val_targets) in enumerate(data_loader.test_loader):
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_outputs = model(val_inputs)
            loss = loss_fn(val_outputs, val_targets)
            val_loss_total += loss.item()
            num_batches_val += 1
            preds = torch.argmax(val_outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(val_targets)

    train_loss_epoch = train_loss_total / num_batches
    val_loss_epoch = val_loss_total / num_batches_val
    train_loss.append(train_loss_epoch)
    val_loss.append(val_loss_epoch)
    
    # Combine all batches for F1 score calculation
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1_score = multiclass_f1_score(all_preds, all_targets, num_classes=num_class)
    f1_score_val = f1_score.item()
    f1_scores.append(f1_score_val)
    
    # Optionally print progress every few epochs
    # if (epoch % 10 == 0) or (epoch % 10 == 5):
    #     print(f"Epoch {epoch}: Train Loss = {train_loss[-1]:.4f}, Val Loss = {val_loss[-1]:.4f}, F1 Score = {f1_score_val*100:.2f}%")
    scheduler.step()

    # 保存当前最佳模型（根据验证集 F1 分数判断）
    if f1_score_val > best_f1:
        best_f1 = f1_score_val
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss_epoch,
            'val_loss': val_loss_epoch,
            'f1_score': f1_score_val,
            # 你还可以加入其他的超参数信息
            'hyperparameters': {
                'lr': lr,
                'batch_size': data_loader.train_loader.batch_size,
                'num_epochs': num_epochs,
                # 其他你认为有用的信息
            }
        }
        torch.save(checkpoint, checkpoint_path)
    

# Plotting the training curves
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot Loss curves in the first subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# Plot F1 score curve in the second subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, f1_scores, label='Validation F1 Score', color='green')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(figure_path)
plt.show()
