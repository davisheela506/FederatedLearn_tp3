import subprocess
import time
from data_utils import generate_distributed_data
import argparse
import os
import random

def run_simulation(alpha: float, algorithm: str = "fedavg", mu: float = 0.1, 
                   attack_type: str = "none", malicious_ratio: float = 0.0):
    # Constants
    NUM_CLIENTS = 10
    BATCH_SIZE = 64
    RESULTS_FILE = f"results_{algorithm}.json"
    SEED = 42

    # Set random seed for reproducibility
    random.seed(SEED)

    # Step 1: Generate data
    print(f"Generating distributed datasets with alpha={alpha}...")
    generate_distributed_data(NUM_CLIENTS, alpha, "./client_data")

    # Step 2: Start server
    print(f"Starting server with {algorithm.upper()}...")
    server_cmd = ["python", "start_server.py", "--algorithm", algorithm]
    if algorithm == "fedprox":
        server_cmd.extend(["--mu", str(mu)])
    server_process = subprocess.Popen(server_cmd)

    # Wait for server to start
    time.sleep(5)

    # Step 3: Assign malicious clients
    print("Starting clients...")
    client_processes = []
    num_malicious = int(NUM_CLIENTS * malicious_ratio)
    malicious_ids = random.sample(range(NUM_CLIENTS), num_malicious)
    print(f"Malicious clients ({attack_type}):", malicious_ids)

    for cid in range(NUM_CLIENTS):
        attack = attack_type if cid in malicious_ids else "none"
        process = subprocess.Popen(["python", "run_client.py", "--cid", str(cid), "--attack_type", attack])
        client_processes.append(process)

    # Step 4: Wait for server to finish
    server_process.wait()

    # Step 5: Terminate clients
    for process in client_processes:
        process.terminate()

    # Step 6: Rename results file
    result_name = f"results_{algorithm}_alpha{alpha}_attack{attack_type}_mal{int(malicious_ratio*100)}.json"
    if os.path.exists(RESULTS_FILE):
        os.rename(RESULTS_FILE, result_name)
    else:
        print(f"Error: Results file '{RESULTS_FILE}' not found.")
        return

    # Step 7: Analyze results
    print("Analyzing results...")
    subprocess.run(["python", "analyze_results.py", result_name])

def main():
    parser = argparse.ArgumentParser(description="Run federated learning simulations for TP3")
    parser.add_argument("--algorithm", type=str, default="fedavg", 
                        choices=["fedavg", "fedprox", "scaffold", "median", "krum"],
                        help="FL aggregation strategy")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet alpha value for data heterogeneity")
    parser.add_argument("--mu", type=float, default=0.1,
                        help="FedProx mu parameter")
    parser.add_argument("--attack_type", type=str, default="none", 
                        choices=["none", "data", "model"],
                        help="Type of client attack: none, data poisoning, or model poisoning")
    parser.add_argument("--malicious_ratio", type=float, default=0.0,
                        help="Proportion of malicious clients (0.0 to 1.0)")
    args = parser.parse_args()

    run_simulation(
        alpha=args.alpha,
        algorithm=args.algorithm,
        mu=args.mu,
        attack_type=args.attack_type,
        malicious_ratio=args.malicious_ratio
    )

if __name__ == "__main__":
    main()
