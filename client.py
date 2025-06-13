# client.py
import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple
from config import FEDPROX_MU, LEARNING_RATE
from model import CustomFashionModel
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters, parameters_to_ndarrays
)

class CustomClient(fl.client.Client):
    def __init__(self, model: CustomFashionModel, train_loader: DataLoader,
                 test_loader: DataLoader, device: torch.device, attack_type: str = "none") -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.attack_type = attack_type.lower()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.global_params = None  # For FedProx
        self.mu = FEDPROX_MU
        self.local_control = None
        self.global_control = None
        self.initial_params = None

    def fit(self, ins: FitIns) -> FitRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        self.initial_params = [torch.from_numpy(p).to(self.device) for p in parameters]
        self.global_params = [torch.from_numpy(p).to(self.device) for p in parameters]

        if "global_control_variate" in ins.config:
            self.global_control = [torch.from_numpy(arr).to(self.device)
                                   for arr in ins.config["global_control_variate"]]
            if self.local_control is None:
                self.local_control = [torch.zeros_like(p).to(self.device)
                                      for p in self.model.parameters()]

        if self.global_control is not None:
            loss, accuracy = self._train_one_epoch_scaffold()
        else:
            loss, accuracy = self._train_one_epoch_fedprox()

        updated_parameters = self.model.get_model_parameters()

        if self.attack_type == "model":
            updated_parameters = [p * 5.0 for p in updated_parameters]  # scale model update

        metrics = {"accuracy": accuracy}
        if self.global_control is not None:
            self._update_local_control_variate()
            metrics["control_variates"] = [cv.cpu().numpy() for cv in self.local_control]

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_parameters),
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )

    def _train_one_epoch_fedprox(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            if self.attack_type == "data":
                target = (target + 1) % 10  # label flipping

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            if self.global_params is not None and self.mu > 0:
                proximal_term = sum((lp - gp).norm(2) for lp, gp in zip(self.model.parameters(), self.global_params))
                loss += (self.mu / 2) * proximal_term

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        return total_loss / total, 100.0 * correct / total

    def _train_one_epoch_scaffold(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            if self.attack_type == "data":
                target = (target + 1) % 10

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            with torch.no_grad():
                for p, lc, gc in zip(self.model.parameters(), self.local_control, self.global_control):
                    if p.grad is not None:
                        p.grad += gc - lc
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        return total_loss / total, 100.0 * correct / total

    def _update_local_control_variate(self) -> None:
        if self.initial_params is None or self.global_control is None:
            return
        current_params = list(self.model.parameters())
        num_batches = len(self.train_loader)

        with torch.no_grad():
            for lc, gc, init, curr in zip(self.local_control, self.global_control, self.initial_params, current_params):
                lc.data = lc - gc + (init - curr) / (LEARNING_RATE * num_batches)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        loss, accuracy = self.model.test_one_epoch(self.test_loader, self.criterion, self.device)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"accuracy": accuracy}
        )
