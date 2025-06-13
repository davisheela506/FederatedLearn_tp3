import flwr as fl
import json
import argparse
from server import (
    CustomClientManager,
    FedAvgStrategy,
    FedProxStrategy,
    SCAFFOLDStrategy,
    FedMedianStrategy,
    KrumStrategy,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "scaffold", "median", "krum"],
                        help="FL algorithm to use")
    parser.add_argument("--mu", type=float, default=0.1,
                        help="Proximal term coefficient for FedProx")
    args = parser.parse_args()

    client_manager = CustomClientManager()

    # Select strategy based on algorithm
    if args.algorithm == "fedavg":
        strategy = FedAvgStrategy()
    elif args.algorithm == "fedprox":
        strategy = FedProxStrategy(mu=args.mu)
    elif args.algorithm == "scaffold":
        strategy = SCAFFOLDStrategy()
    elif args.algorithm == "median":
        strategy = FedMedianStrategy()
    elif args.algorithm == "krum":
        strategy = KrumStrategy(f=5)  # Assumes up to 5 malicious clients

    print(f"[Server] Starting Federated Server with {args.algorithm.upper()} strategy...")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
        client_manager=client_manager
    )

    # Save results
    results = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed
    }
    with open(f"results_{args.algorithm}.json", "w") as f:
        json.dump(results, f)

    print(f"[Server] Training complete. Results saved to results_{args.algorithm}.json")

if __name__ == "__main__":
    main()
