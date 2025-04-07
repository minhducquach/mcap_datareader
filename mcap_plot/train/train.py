import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import itertools
from dataset import CustomDataset
from model import MLP
from utils import alternate_training, evaluate_model

train_files = ["../txt/light_tab_dis_data_fb.txt",
                  "../txt/light_tab_dis_data_lr.txt",
                  "../txt/light_tab_dis_data_ud.txt"]
test_files = ["../txt/light_tab_dis_data_diffax.txt"]

train_datasets = {file: CustomDataset(file) for file in train_files}
train_loaders = {file: (DataLoader(ds, batch_size=32, shuffle=True)) for file, ds in train_datasets.items()}

test_datasets = {file: CustomDataset(file) for file in test_files}
test_loaders = {file: (DataLoader(ds, batch_size=32, shuffle=False)) for file, ds in test_datasets.items()}

input_size = 3
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

model = alternate_training(train_loaders, model, criterion, optimizer, num_epochs=100)
test_results = evaluate_model(model, test_loaders)

torch.save(model.state_dict(), 'trained_model.pth')
