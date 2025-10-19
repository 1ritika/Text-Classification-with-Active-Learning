import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
 
 
def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"
    
    # We'll store the arrays for each run here
    all_active_runs = []
    all_random_runs = []
 
    # We'll collect times for each strategy across runs
    all_times_no = []  # stores training times for "no"
    all_times_yes = []   # stores training times for "yes"
 
    # For data sizes, we assume they match across runs,
    # but we can store them too
    all_sizes_no = []  # stores data sizes for "no"
    all_sizes_yes = []   # stores data sizes for "yes"
 
    for run_id in range(1, 6):
        # Build file paths
        no_path = os.path.join(args.logs_path, f"run_{run_id}_True_no_times.npy")
        yes_path = os.path.join(args.logs_path, f"run_{run_id}_True_yes_times.npy")
 
        no_size_path = os.path.join(args.logs_path, f"run_{run_id}_True_no_sizes.npy")
        yes_size_path = os.path.join(args.logs_path, f"run_{run_id}_True_yes_sizes.npy")
 
        # Check files exist
        if not os.path.exists(no_path):
            raise FileNotFoundError(f"Missing file: {no_path}")
        if not os.path.exists(yes_path):
            raise FileNotFoundError(f"Missing file: {yes_path}")
        if not os.path.exists(no_size_path):
            raise FileNotFoundError(f"Missing file: {no_size_path}")
        if not os.path.exists(yes_size_path):
            raise FileNotFoundError(f"Missing file: {yes_size_path}")
 
        # Load times
        times_no = np.load(no_path)  # shape: (#iterations,)
        times_yes = np.load(yes_path)
        # Load data sizes
        sizes_no = np.load(no_size_path)  # shape: (#iterations,)
        sizes_yes = np.load(yes_size_path)
 
        # Check length
        if len(times_no) != len(times_yes):
            raise ValueError(f"Mismatch in iteration length for run_id={run_id} between no vs yes.")
        # Also check that sizes match times in shape
        if len(times_no) != len(sizes_no):
            raise ValueError(f"Mismatch: times vs sizes for no run_id={run_id}.")
        if len(times_yes) != len(sizes_yes):
            raise ValueError(f"Mismatch: times vs sizes for yes run_id={run_id}.")
 
        all_times_no.append(times_no)
        all_times_yes.append(times_yes)
        all_sizes_no.append(sizes_no)
        all_sizes_yes.append(sizes_yes)
 
    # Convert to arrays: shape => (5, #iterations)
    all_times_no = np.array(all_times_no)
    all_times_yes = np.array(all_times_yes)
    all_sizes_no = np.array(all_sizes_no)
    all_sizes_yes = np.array(all_sizes_yes)
 
    # Compute mean, std for times
    mean_times_no = all_times_no.mean(axis=0)
    std_times_no = all_times_no.std(axis=0)
    mean_times_yes = all_times_yes.mean(axis=0)
    std_times_yes = all_times_yes.std(axis=0)
 
    # If you want to plot times vs. iteration index:
    x_axis = np.arange(len(mean_times_no))  # iteration indices
 
    plt.figure(figsize=(8,5))
 
    # Plot no times
    plt.plot(x_axis, mean_times_no, color="green", label="Without update (mean)")
    plt.fill_between(x_axis,
                     mean_times_no - std_times_no,
                     mean_times_no + std_times_no,
                     alpha=0.2, color="green")
 
    # Plot yes times
    plt.plot(x_axis, mean_times_yes, color="purple", label="With update (mean)")
    plt.fill_between(x_axis,
                     mean_times_yes - std_times_yes,
                     mean_times_yes + std_times_yes,
                     alpha=0.2, color="purple")
 
    plt.xlabel("Iteration indices")
    plt.ylabel("Training time (in seconds)")
    plt.title("Comparison of training times: with update vs without update")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.logs_path, "compare_times.png"))
    print(f"Saved time comparison plot to {os.path.join(args.logs_path, 'compare_times.png')}")

    # Load the runs from run_1 to run_5
    for run_id in range(1, 6):
        path_active = os.path.join(args.logs_path, f"run_{run_id}_True.npy")
        path_random = os.path.join(args.logs_path, f"run_{run_id}_False.npy")
 
        # Check if these files exist
        if not os.path.exists(path_active):
            raise FileNotFoundError(f"Missing file: {path_active}")
        if not os.path.exists(path_random):
            raise FileNotFoundError(f"Missing file: {path_random}")
 
        acc_active = np.load(path_active)   # shape: (#iterations,)
        acc_random = np.load(path_random)   # shape: (#iterations,)
 
        # Check they match in length
        if len(acc_active) != len(acc_random):
            raise ValueError(
                f"Mismatch in lengths for run_id={run_id}: "
                f"active={len(acc_active)} vs random={len(acc_random)}"
            )
 
        all_active_runs.append(acc_active)
        all_random_runs.append(acc_random)
 
    # Convert to arrays: shape = (5, #iterations)
    all_active_runs = np.array(all_active_runs)
    all_random_runs = np.array(all_random_runs)
 
    # Check that all runs produce the same number of iterations
    # (should already be consistent from the above check)
    num_iterations = all_active_runs.shape[1]
 
    # Compute means and stds across the 5 runs
    mean_active = all_active_runs.mean(axis=0)
    std_active = all_active_runs.std(axis=0)
    mean_random = all_random_runs.mean(axis=0)
    std_random = all_random_runs.std(axis=0)
 
    # x-axis: iteration index
    x_axis = np.arange(num_iterations)
 
    # Create a figure
    plt.figure(figsize=(8,5))
 
    # Plot Active Learning curve
    plt.plot(x_axis, mean_active, label="Active Learning", color="red")
    # Fill between mean ± std
    plt.fill_between(x_axis, mean_active - std_active, mean_active + std_active,
                     alpha=0.2, color="red")
 
    # Plot Random Sampling curve
    plt.plot(x_axis, mean_random, label="Random Sampling", color="cyan")
    plt.fill_between(x_axis, mean_random - std_random, mean_random + std_random,
                     alpha=0.2, color="cyan")
 
    # Horizontal line for fully supervised accuracy
    plt.axhline(y=args.supervised_accuracy, color="black", linestyle="--",
                label="Fully Supervised Learning")
 
    plt.xlabel("Iteration (each: +4500 samples)")
    plt.ylabel("Validation Accuracy")
    plt.title("Active Learning vs Random Sampling")
    plt.legend(loc="lower right")
 
    # Either show or save
    plt.tight_layout()
    plt.savefig(os.path.join(args.logs_path, "active_vs_random.png"))
    print(f"Plot saved to {os.path.join(args.logs_path, 'active_vs_random.png')}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())
