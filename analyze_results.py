import json
import sys
import matplotlib.pyplot as plt
import os

def load_results(filename: str):
    if not os.path.exists(filename):
        print(f"Error: Results file '{filename}' not found.")
        return None
    with open(filename, "r") as f:
        return json.load(f)

def plot_metrics(results, title_prefix=""):
    rounds = list(range(1, len(results["losses_distributed"]) + 1))

    # Extract loss values
    losses = [entry[1] for entry in results["losses_distributed"]]

    # Extract accuracy values
    metrics = results.get("metrics_distributed", {}).get("accuracy", [])
    accuracies = [acc[1] for acc in metrics] if metrics else [0.0] * len(rounds)

    # Print metrics to terminal
    print("+-------+--------+----------+")
    print("| Round |  Loss  | Accuracy |")
    print("+-------+--------+----------+")
    for r, l, a in zip(rounds, losses, accuracies):
        print(f"| {r:5d} | {l:6.4f} | {a:8.4f} |")
    print("+-------+--------+----------+")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_title(f"{title_prefix} Training Loss and Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(rounds, losses, color="tab:red", label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(rounds, accuracies, color="tab:blue", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    plt.grid(True)

    # Save the plot
    output_file = f"{title_prefix}_plot.png"
    plt.savefig(output_file)
    print(f"[INFO] Plot saved as: {output_file}")

    # Optional: uncomment to view interactively if GUI is available
    # plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        return

    filename = sys.argv[1]
    results = load_results(filename)
    if results is not None:
        title_prefix = os.path.splitext(os.path.basename(filename))[0]
        plot_metrics(results, title_prefix=title_prefix)

if __name__ == "__main__":
    main()
