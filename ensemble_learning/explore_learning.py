from architecture import arch1, arch2, arch3, arch4, arch5, arch6
from utils import (
    load_h5,
    report_prediction,
    init_logger,
    dump_pkl,
    write_report,
)
from data_preparation import prepare_data_cnn

from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from datetime import datetime


def logg_info_data(
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    le,
    frac_test,
    frac_val,
    frac_false,
    n_epochs,
    batch_size,
    categorical,
    logg,
):
    """Log information about the data

    Parameters
    ----------
    X_train : numpy array
        Input data (features) for training

    X_test : numpy array
        Input data (features) for testing

    X_val : numpy array
        Input data (features) for validation

    y_train : numpy array
        Labels of the data (str or int) for training

    y_test : numpy array
        Labels of the data (str or int) for testing

    y_val : numpy array
        Labels of the data (str or int) for validation

    le : sklearn.preprocessing.LabelEncoder
        LabelEncoder used to encode the labels

    frac_test : float
        Fraction of the data to use for testing

    frac_val : float
        Fraction of the data to use for validation

    frac_false : float
        Percentage of false labels to generate

    n_epochs : int
        Number of epochs to train the model

    batch_size : int
        Batch size to use for training

    categorical : bool
        If True, encode the labels using to_categorical

    logg : logging
        Logger

    Returns
    -------
    logging
        Logger with the information about the data
    """
    logg.info("Data loaded")
    logg.info("Preparing data")
    logg.info(f"fraction test : {frac_test}")
    logg.info(f"fraction validation : {frac_val}")
    logg.info(f"fraction false : {frac_false}")
    logg.info(f"categorical : {categorical}")
    logg.info(f"X_train shape : {X_train.shape}")
    logg.info(f"X_test shape : {X_test.shape}")
    logg.info(f"X_val shape : {X_val.shape}")
    logg.info(f"y_train : {np.unique(y_train, return_counts=True)}")
    logg.info(f"y_test : {np.unique(y_test, return_counts=True)}")
    logg.info(f"y_val : {np.unique(y_val, return_counts=True)}")
    logg.info(f"le.classes_ : {le.classes_}")
    logg.info(f"le.transform(le.classes_) : {le.transform(le.classes_)}")
    logg.info(f"n_epochs : {n_epochs}")
    logg.info(f"batch_size : {batch_size}")
    return logg


def extract_best_lr(dic):
    """Extract the best learning rate for each model stored in a dictionary

    Parameters
    ----------
    dic : dict
        Dictionary with the f1 and accuracy for each model and learning rate

    Returns
    -------
    pandas.DataFrame
        Dataframe with the best learning rate for each model
    """

    pdf = pd.DataFrame(dic).T
    pdf.columns = ["f1", "acc"]
    pdf["learning_rate"] = pdf.index.str.split("_").str[1].astype(float)
    pdf["model"] = pdf.index.str.split("_").str[0]
    pdf.reset_index(drop=True, inplace=True)
    best_lr_per_model = pdf.groupby("model").apply(lambda x: x.loc[x["f1"].idxmax()])
    return best_lr_per_model


def info_training(m, h, logg, dic):
    """Log information about the training of a keras model, calculate the f1 and accuracy and store them in a dictionary.
    Log the first and last loss and val_loss

    Parameters
    ----------
    m : keras model
        Model to train

    h : keras history
        History of the training

    logg : logging
        Logger

    dic : dict
        Dictionary with the f1 and accuracy for each model and learning rate

    Returns
    -------
    logging
        Logger with the information about the training
    """
    logg.info(f"Model trained")
    top = {np.array(h.history["loss"])[0].round(5)}
    end = {np.array(h.history["loss"])[-1].round(5)}
    logg.info(f"Loss:  {top} -> {end}")
    top_val = {np.array(h.history["val_loss"])[0].round(5)}
    end_val = {np.array(h.history["val_loss"])[-1].round(5)}
    logg.info(f"Val loss:  {top_val} -> {end_val}")
    ypred = m.predict(X_test)
    logg, f1, acc = report_prediction(y_test, ypred, le, logg)
    dic[name_model] = [f1, acc]
    return logg, dic


if __name__ == "__main__":
    frac_false = -1
    frac_test = 0.2
    frac_val = 0.15
    categorical = True
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
    ]
    n_epochs = 750
    batch_size = 4096

    name_folder = datetime.now().strftime("%d%m%y_%HH%MM%S")
    path_results = f"../data/results/{name_folder}/"
    path_data = "../data/divergence_process.h5"
    os.makedirs(path_results, exist_ok=True)
    logg, pathlog = init_logger(path_results)

    logg.info("Loading data")
    logg.info(f"Path data : {path_data}")

    X, y = load_h5(path_data)

    X_train, X_test, X_val, y_train, y_test, y_val, le = prepare_data_cnn(
        X,
        y,
        frac_test=frac_test,
        frac_val=frac_val,
        frac_false=frac_false,
        categorical=categorical,
    )

    logg = logg_info_data(
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        le,
        frac_test,
        frac_val,
        frac_false,
        n_epochs,
        batch_size,
        categorical,
        logg,
    )
    dic = {}
    lr = [0.15, 0.1, 0.075, 0.05, 0.0375, 0.025, 0.02]
    models = [arch1, arch2, arch3, arch4, arch5, arch6]
    for model in tqdm(models, desc="model", leave=False):
        for learning_rate in tqdm(lr, desc="learning rate", leave=False):
            m = model(learning_rate, num_classes=y_train.shape[1])
            name_model = f"{m.name}_{learning_rate}"
            logg.info("#" * 50)
            logg.info(f"Learning rate : {learning_rate}")
            m.summary(print_fn=lambda x: logg.info(x))
            h = m.fit(
                X_train,
                y_train,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                workers=4,
                use_multiprocessing=True,
                verbose=0,
            )
            logg, dic = info_training(m, h, logg, dic)
            del m, h
    dump_pkl(dic, os.path.join(path_results, "dic.pkl"))

    best_lr_per_model = extract_best_lr(dic)
    logg.info(f"best_lr_per_model : \n{best_lr_per_model}")

    pathreport = os.path.join(path_results, "report.txt")
    write_report(pathlog, pathreport)
