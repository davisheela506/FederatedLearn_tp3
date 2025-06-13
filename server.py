import flwr as fl
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, EvaluateIns, Scalar, FitRes, EvaluateRes
from typing import List, Tuple, Dict, Optional, Union
import threading
import numpy as np
import torch
from model import CustomFashionModel
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from scipy.spatial.distance import cdist


# -------------------------
# Custom Client Manager
# -------------------------
class CustomClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = threading.Lock()

    def num_available(self) -> int:
        with self.lock:
            return len(self.clients)

    def register(self, client: ClientProxy) -> bool:
        with self.lock:
            if client.cid in self.clients:
                return False
            self.clients[client.cid] = client
            return True

    def unregister(self, client: ClientProxy) -> None:
        with self.lock:
            self.clients.pop(client.cid, None)

    def all(self) -> Dict[str, ClientProxy]:
        with self.lock:
            return self.clients.copy()

    def wait_for(self, num_clients: int, timeout: int) -> bool:
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if len(self.clients) >= num_clients:
                    return True
            time.sleep(1)
        return False

    def sample(self, num_clients: int, min_num_clients: Optional[int] = None,
               criterion: Optional[object] = None) -> List[ClientProxy]:
        min_num_clients = min_num_clients or num_clients
        if not self.wait_for(min_num_clients, timeout=60):
            return []
        with self.lock:
            import random
            clients = list(self.clients.values())
            return random.sample(clients, min(num_clients, len(clients)))


# -------------------------
# FedAvg Strategy
# -------------------------
class FedAvgStrategy(Strategy):
    def __init__(self):
        self.model = CustomFashionModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return parameters

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_weights = self._aggregate(weights_results)
        parameters = ndarrays_to_parameters(aggregated_weights)

        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0

        return parameters, {"accuracy": aggregated_accuracy}

    def _aggregate(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        num_examples_total = sum(num_examples for _, num_examples in results)
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]
        aggregated_weights = [
            np.sum(np.stack([w[i] for w in weighted_weights]), axis=0) / num_examples_total
            for i in range(len(weighted_weights[0]))
        ]
        return aggregated_weights

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        evaluate_ins = EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        losses = [res.loss * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0.0

        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0

        return float(aggregated_loss), {"accuracy": aggregated_accuracy}

    def evaluate(self, server_round, parameters):
        return None


# -------------------------
# FedProx Strategy
# -------------------------
class FedProxStrategy(FedAvgStrategy):
    def __init__(self, mu: float = 0.1):
        super().__init__()
        self.mu = mu

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        fit_ins = FitIns(parameters, {"mu": self.mu})
        return [(client, fit_ins) for client in clients]


# -------------------------
# SCAFFOLD Strategy
# -------------------------
class SCAFFOLDStrategy(FedAvgStrategy):
    def __init__(self):
        super().__init__()
        self.global_control_variate = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        if self.global_control_variate is None:
            self.global_control_variate = [torch.zeros_like(p) for p in self.model.parameters()]
        return parameters

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        cv_np = [cv.cpu().numpy() for cv in self.global_control_variate]
        fit_ins = FitIns(parameters, {"global_control_variate": cv_np})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_weights = self._aggregate(weights_results)
        parameters = ndarrays_to_parameters(aggregated_weights)

        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0

        if "control_variates" in results[0][1].metrics:
            cv_updates = [res.metrics["control_variates"] for _, res in results]
            num_clients = len(results)

            cv_updates_tensors = []
            for client_cv in cv_updates:
                cv_updates_tensors.append([torch.from_numpy(arr).to(self.device) for arr in client_cv])

            for i in range(len(self.global_control_variate)):
                update_sum = torch.zeros_like(self.global_control_variate[i])
                for client_cv in cv_updates_tensors:
                    update_sum += client_cv[i]
                self.global_control_variate[i] += update_sum / num_clients

        return parameters, {"accuracy": aggregated_accuracy}


# -------------------------
# FedMedian Strategy
# -------------------------
class FedMedianStrategy(FedAvgStrategy):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        num_params = len(all_updates[0])

        median_weights = []
        for i in range(num_params):
            stacked = np.stack([update[i] for update in all_updates], axis=0)
            median = np.median(stacked, axis=0)
            median_weights.append(median)

        parameters = ndarrays_to_parameters(median_weights)

        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0

        return parameters, {"accuracy": aggregated_accuracy}


# -------------------------
# Krum Strategy
# -------------------------
class KrumStrategy(FedAvgStrategy):
    def __init__(self, f=2):
        super().__init__()
        self.f = f  # assumed number of attackers

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        num_clients = len(updates)
        flat_updates = [np.concatenate([u.flatten() for u in update]) for update in updates]

        distances = cdist(flat_updates, flat_updates, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])[:num_clients - self.f - 2]
            scores.append(np.sum(sorted_distances))

        best_idx = int(np.argmin(scores))
        selected_update = updates[best_idx]
        parameters = ndarrays_to_parameters(selected_update)

        return parameters, {"accuracy": results[best_idx][1].metrics["accuracy"]}
