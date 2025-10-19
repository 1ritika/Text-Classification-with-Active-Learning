import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp
from collections import Counter


class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self, max_vocab_len=10_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None  # stores vocabulary mapping word to index
        self.max_vocab_len = max_vocab_len  # max words to keep in vocabulary
        self.ngram_range = (1,3)  # n grams are used e.g.- (1,3) means unigrams, bigrams, trigrams
        self.lambdaweights = (0.4,0.4,0.2)  # weight assigned to different n-gram levels

        # TODO: Add more class variables if needed

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        # TODO: count the occurrences of each word
        # TODO: sort the words based on frequency
        # TODO: retain the top 10k words

        ngram_counter = Counter()  # keeps track of n-gram occurrences

        for text in X_train:
            word_list = text.split()  # tokenize sentence into words
            # Loop over n-grams in the specified range
            for idx, ngram_size in enumerate(range(self.ngram_range[0], self.ngram_range[1]+1)):
                weight_factor = self.lambdaweights[idx]  # assign weight based on n-gram level
                if len(word_list) < ngram_size:
                    continue
                # Compute n-grams for this sentence
                ngram_list = [" ".join(word_list[i:i+ngram_size]) for i in range(len(word_list)-ngram_size+1)]
                for gram in ngram_list:
                    ngram_counter[gram] += weight_factor  # update frequency with weight

        for gram in list(ngram_counter.keys()):
            if ngram_counter[gram] < 2:  # remove rare words (occurring < 2 times)
                del ngram_counter[gram]

        # Retain the top max_vocab_len words
        most_common_ngrams = ngram_counter.most_common(self.max_vocab_len)
        # Build vocabulary as a dict mapping word -> index.
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common_ngrams)}

    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"

        row_indices = []  # stores row indices for sparse matrix
        col_indices = []  # stores column indices for sparse matrix
        values = []  # stores values for sparse matrix

        for doc_idx, text in enumerate(X):
            word_list = text.split()  # tokenize sentence into words
            # For each n-gram order
            for idx, ngram_size in enumerate(range(self.ngram_range[0], self.ngram_range[1]+1)):
                weight_factor = self.lambdaweights[idx]  # weight for this n-gram level
                if len(word_list) < ngram_size:
                    continue
                ngram_list = [" ".join(word_list[j:j+ngram_size]) for j in range(len(word_list)-ngram_size+1)]
                ngram_counts = Counter(ngram_list)  # count occurrences of n-grams
                for gram, count in ngram_counts.items():
                    if gram in self.vocab:
                        row_indices.append(doc_idx)  # add row index
                        col_indices.append(self.vocab[gram])  # add column index
                        values.append(count)  # add count value

        n_samples = len(X)  # total samples
        vocab_size = len(self.vocab)  # vocabulary size
        # Create a sparse CSR matrix from the data.
        X_sparse = sp.csr_matrix((values, (row_indices, col_indices)), shape=(n_samples, vocab_size))
        return X_sparse


def set_seed(seed: int):
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)  # set seed for random module
    np.random.seed(seed)  # set seed for numpy


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')  # load csv file into dataframe

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle dataset

    # split into train, val and test set
    train_size = int(0.8 * len(df))  # 80% for training, 20% for validation
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train = \
        train_df['stemmed_content'].values, train_df['target'].values  # extract train data
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values  # extract validation data
    return x_train, y_train, x_val, y_val
