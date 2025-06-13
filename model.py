import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import List

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        return total_loss / total, 100.0 * correct / total

    def test_one_epoch(self, test_loader, criterion, device):
        self.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return total_loss / total, 100.0 * correct / total

    def get_model_parameters(self) -> List[np.ndarray]:
        return [param.detach().cpu().numpy() for param in self.parameters()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(new_param).to(param.device)
