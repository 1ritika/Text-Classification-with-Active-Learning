import os
import time
import pickle
import numpy as np
import argparse
from utils import set_seed, get_data
from model import MultinomialNaiveBayes

def softmax(logits):
    """
    logits: np.ndarray of shape (num_samples, num_classes)
    returns probabilities of shape (num_samples, num_classes)
    """
    # For numerical stability, subtract the max along each row
    # so that exp(...) never gets too large
    adjusted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(adjusted_logits)  # apply exponentiation
    sum_exp = np.sum(exp_values, axis=1, keepdims=True)  # sum of exponentials
    return exp_values / sum_exp  # return softmax probabilities

def main(args: argparse.Namespace):
    # set seed for reproducibility
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no + args.run_id)  # ensure reproducibility

    # Load the preprocessed data
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_matrix = pickle.load(open(f"{args.data_path}/X_train{args.intermediate}", "rb"))
        X_val_matrix = pickle.load(open(f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_train_labels = pickle.load(open(f"{args.data_path}/y_train{args.intermediate}", "rb"))
        y_val_labels = pickle.load(open(f"{args.data_path}/y_val{args.intermediate}", "rb"))

        # Shuffle training data
        shuffled_indices = np.random.RandomState(args.run_id).permutation(X_train_matrix.shape[0])
        X_train_matrix = X_train_matrix[shuffled_indices]
        y_train_labels = y_train_labels[shuffled_indices]
        print("Preprocessed Data Loaded")
    else:
        raise Exception("Preprocessed Data not found")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    validation_accuracies = []  # store validation accuracy per iteration
    training_accuracies = []  # store training accuracy per iteration
    total_training_samples = 10_000
    selected_indices = np.arange(10_000)
    remaining_indices = np.setdiff1d(np.arange(X_train_matrix.shape[0]), selected_indices)

    # Track training time and dataset sizes per iteration
    training_times = []
    dataset_sizes = []

    for iteration in range(1, 60):
        dataset_sizes.append(len(selected_indices))  # track current dataset size

        # Start timing
        start_time = time.time()

        if args.update == "no":
            # Train from scratch on the currently labeled data
            X_train_subset = X_train_matrix[selected_indices]
            y_train_subset = y_train_labels[selected_indices]
            model.fit(X_train_subset, y_train_subset, update=False)

        elif args.update == "yes":
            # If first iteration, train from scratch
            if iteration == 1:
                X_initial = X_train_matrix[selected_indices]
                y_initial = y_train_labels[selected_indices]
                model.fit(X_initial, y_initial, update=False)
            else:
                # Update model incrementally with newly selected data
                X_new_samples = X_train_matrix[newly_selected]
                y_new_samples = y_train_labels[newly_selected]
                model.fit(X_new_samples, y_new_samples, update=True)
        else:
            raise ValueError("Invalid update value. Must be 'yes' or 'no'.")

        # End timing
        end_time = time.time()
        training_times.append(end_time - start_time)

        # Evaluate model on validation set
        y_val_predictions = model.predict(X_val_matrix)
        validation_accuracy = np.mean(y_val_predictions == y_val_labels)
        print(f"{len(selected_indices)} items - Validation acc: {validation_accuracy}")
        validation_accuracies.append(validation_accuracy)

        # Compute training accuracy on the current dataset
        y_train_predictions = model.predict(X_train_matrix[selected_indices])
        training_accuracy = np.mean(y_train_predictions == y_train_labels[selected_indices])
        print(f"{len(selected_indices)} items - Training acc: {training_accuracy}")
        training_accuracies.append(training_accuracy)

        # Select new samples based on active learning or random selection
        if args.is_active:
            log_prior_probs = np.log(model.priors + 1e-12)
            log_likelihoods = np.log(model.means + 1e-12)
            X_unlabeled_pool = X_train_matrix[remaining_indices]
            log_probabilities = X_unlabeled_pool.dot(log_likelihoods.T) + log_prior_probs  # shape: [num_pool, num_classes]
            probability_distribution = softmax(log_probabilities)  # convert to probabilities

            # For binary classification, compute margin: lower margin => higher uncertainty
            uncertainty_margins = np.abs(probability_distribution[:, 0] - probability_distribution[:, 1])
            uncertain_order = np.argsort(uncertainty_margins)  # sort by uncertainty

            # Select most uncertain samples
            num_selected = min(4500, len(uncertain_order))
            newly_selected = remaining_indices[uncertain_order[:num_selected]]

            # Add selected samples to training set
            selected_indices = np.concatenate([selected_indices, newly_selected])
            # Remove from the remaining pool
            remaining_indices = np.setdiff1d(remaining_indices, newly_selected)
        else:
            newly_selected = remaining_indices[:4500]
            selected_indices = np.concatenate([selected_indices, newly_selected])
            remaining_indices = remaining_indices[4500:]
        total_training_samples += 4500

    # Save validation accuracy results
    validation_accuracies = np.array(validation_accuracies)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", validation_accuracies)

    # Save training accuracy results
    training_accuracies = np.array(training_accuracies)
    np.save(f"{args.logs_path}/train_{args.run_id}_{args.is_active}_{args.update}.npy", training_accuracies)

    # Save training times and dataset sizes
    training_times = np.array(training_times)
    dataset_sizes = np.array(dataset_sizes)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}_{args.update}_times.npy", training_times)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}_{args.update}_sizes.npy", dataset_sizes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=10000)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--update", type=str, required=True,
                        help="Either 'yes' or 'no'")
    main(parser.parse_args())
