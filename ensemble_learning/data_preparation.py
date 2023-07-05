import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def generate_random_labels(labels, false_percentage):
    """Generate random labels from the true labels, used for multi-label classification

    Note
    ----
    The current implementation is not optimized and give a percentage of false labels that is not exactly equal to false_percentage

    Parameters
    ----------
    labels : numpy array
        True labels (1D array of int or str)

    false_percentage : float
        Percentage of false labels to generate

    Returns
    -------
    numpy array
        Random labels
    """

    unique_labels = np.unique(labels)
    num_false_labels = int(len(labels) * false_percentage)
    idx = np.arange(len(labels))

    np.random.shuffle(idx)
    idx = idx[:num_false_labels]
    labels_copy = labels.copy()
    proba = np.unique(labels_copy[idx], return_counts=True)[1] / len(labels_copy[idx])
    for i in tqdm(idx):
        unqlbl = np.where(unique_labels != labels_copy[i])[0]
        prob = proba[unqlbl]
        prob = prob / np.sum(prob)
        labels_copy[i] = np.random.choice(
            unique_labels[unqlbl], size=1, replace=False, p=prob
        )[0]
    return labels_copy


def prepare_data_cnn(
    X,
    y,
    frac_test=0.2,
    frac_val=0.15,
    frac_false=-1,
    categorical=True,
    idx_train=None,
    idx_test=None,
):
    """Prepare the data for training a CNN model by splitting the data into train, test and validation, encode the labels and generate random labels if frac_false > 0

    Parameters
    ----------
    X : numpy array
        Input data (features)

    y : numpy array
        Labels of the data (str or int, 1D array)

    frac_test : float, optional
        Fraction of the data to use for testing, if idx_train and idx_test are not given, by default 0.2

    frac_val : float, optional
        Fraction of the data to use for validation, by default 0.15

    frac_false : float, optional
        Percentage of false labels to generate, by default -1

    categorical : bool, optional
        If True, encode the labels using to_categorical, by default True

    idx_train : numpy array, optional
        Indices of the data to use for training, necessary if a specific split is used, by default None

    idx_test : numpy array, optional
        Indices of the data to use for testing, necessary if a specific split is used, by default None

    Returns
    -------
    numpy array
        data for the training set

    numpy array
        data for the test set

    numpy array
        data for the validation set

    numpy array
        labels for the training set

    numpy array
        labels for the test set

    numpy array
        labels for the validation set

    LabelEncoder
        LabelEncoder used to encode the labels

    """
    le = LabelEncoder()
    y = le.fit_transform(y)
    if idx_train is not None:
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac_test)
    if frac_false > 0:
        y_train = generate_random_labels(y_train, frac_false)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=frac_val
    )
    if categorical:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val, le


def count_sample_different_per_classe(y, yn):
    """Count the number of samples that are different from the true label per class

    Parameters
    ----------
    y : numpy array
        True labels

    yn : numpy array
        Noisy labels

    Returns
    -------
    None
    """
    unique_labels = np.unique(y)
    for i in unique_labels:
        print(
            f"{i} : ",
            np.sum(yn[np.where(y == i)[0]] != y[np.where(y == i)[0]]) / np.sum(y == i),
        )
