import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
 
def main(args: argparse.Namespace):
    # Build file paths for training and validation accuracy.
    train_path = os.path.join(args.logs_path, f"train_{args.run_id}_True_{args.update}.npy")
    val_path   = os.path.join(args.logs_path, f"run_{args.run_id}_True.npy")
    
    # Check if files exist.
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training accuracy file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation accuracy file not found: {val_path}")
    
    # Load accuracy data.
    train_acc = np.load(train_path)  # Training accuracy per iteration.
    val_acc = np.load(val_path)      # Validation accuracy per iteration.
    
    # Ensure arrays have the same length.
    if len(train_acc) != len(val_acc):
        raise ValueError("Mismatch in the number of iterations between training and validation accuracy.")
    
    iterations = np.arange(len(train_acc))
    
    # Create the plot.
    plt.figure(figsize=(8,5))
    plt.plot(iterations, train_acc, label="Training Accuracy", color="green")
    plt.plot(iterations, val_acc, label="Validation Accuracy", color="blue")
    plt.xlabel("Iteration (each iteration = +4500 samples)")
    plt.ylabel("Accuracy")
    plt.title(f"Training vs. Validation Accuracy (Run {args.run_id}, Active Learning Update)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the plot.
    save_path = os.path.join(args.logs_path, f"train_vs_val_run_{args.run_id}_True_{args.update}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True, help="Run id (1-5)")
    parser.add_argument("--logs_path", type=str, default="logs", help="Path to logs folder")
    parser.add_argument("--update", type=str, required=True,
                        help="Update strategy used (e.g., 'yes' for incremental, 'no' for full retrain)")
    args = parser.parse_args()
    main(args)
