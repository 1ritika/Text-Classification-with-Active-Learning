import numpy as np
from scipy import sparse as sp


class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes model
    """
    def __init__(self, alpha=0.01) -> None:
        """
        Initialize the model
        :param alpha: float
            The Laplace smoothing factor (used to handle 0 probs)
            Hint: add this factor to the numerator and denominator
        """
        self.alpha = alpha  # smoothing factor to avoid zero probabilities
        self.priors = None  # stores class priors
        self.means = None  # stores conditional probabilities
        self.classes = None  # unique class labels
        self.example_count = 0  # to keep track of the number of samples seen

        # NEW: store running counts for partial updates
        self.class_totals = None  # shape: [n_classes], stores count of each class
        self.feature_counts = None  # shape: [n_classes, n_features], stores word counts per class
        self.total_samples = 0  # track total number of samples seen

    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model on the training data
        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether the model is being updated with new data
            or trained from scratch
        :return: None
        """
        if not update:
            # initialize classes
            self.classes = np.unique(y)
            num_classes = len(self.classes)
            num_features = X.shape[1]

            # initialize class and feature counts from scratch
            self.class_totals = np.zeros(num_classes, dtype=np.float64)
            self.feature_counts = np.zeros((num_classes, num_features), dtype=np.float64)
            self.total_samples = 0  # reset total samples

        else:
            # validate if model was previously trained
            if self.classes is None:
                raise ValueError("Model not previously fit; 'update=True' is invalid.")

            num_classes = len(self.classes)
            num_features = X.shape[1]
            if self.feature_counts.shape[1] != num_features:
                raise ValueError("Feature dimension mismatch in partial update.")

        # 1. Update class_totals and feature_counts with new data
        for idx, label in enumerate(self.classes):
            mask = (y == label)  # select samples belonging to the current class
            class_count_batch = np.sum(mask)  # count of samples in this class
            self.class_totals[idx] += class_count_batch  # update class count

            # sum feature counts for class
            if class_count_batch > 0:
                self.feature_counts[idx, :] += X[mask].sum(axis=0).A1  # update feature count

        # update total number of samples seen
        self.total_samples += X.shape[0]

        # 2. Compute class priors
        #    prior(c) = class_totals[c] / total_samples
        total_samples = np.sum(self.class_totals)
        self.priors = self.class_totals / total_samples

        # 3. Compute likelihood probabilities
        #    means[c, f] = (feature_counts[c,f] + alpha) / ( sum_c + alpha * num_features )
        class_sums = np.sum(self.feature_counts, axis=1, keepdims=True)  # sum of features per class
        self.means = (self.feature_counts + self.alpha) / (class_sums + self.alpha * num_features)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            The predicted labels
        """
        assert self.priors.shape[0] == self.means.shape[0]
        assert self.priors is not None and self.means is not None, "Model must be fit before prediction."

        # compute log priors to avoid underflow issues
        log_prior_probs = np.log(self.priors + 1e-12)  # small value added for numerical stability
        log_likelihoods = np.log(self.means + 1e-12)  # compute log probabilities

        # compute log probabilities for each class using matrix multiplication
        log_probs = X.dot(log_likelihoods.T)  # shape: (n_samples, n_classes)
        log_probs += log_prior_probs  # add prior log probabilities

        # select class with highest probability
        predicted_indices = np.argmax(log_probs, axis=1)  # get index of max log probability
        predicted_labels = self.classes[predicted_indices]  # map indices back to class labels
        return predicted_labels
