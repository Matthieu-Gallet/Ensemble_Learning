from architecture import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

from utils_II import *


def generate_random_labels(labels, false_percentage):
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
    X, y, frac_test=0.2, frac_val=0.15, frac_false=-1, categorical=True
):
    le = LabelEncoder()
    y = le.fit_transform(y)
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
    unique_labels = np.unique(y)
    for i in unique_labels:
        print(
            f"{i} : ",
            np.sum(yn[np.where(y == i)[0]] != y[np.where(y == i)[0]]) / np.sum(y == i),
        )


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


def report_prediction(y_true, y_pred, le, logg):
    logg.info("----------- REPORT -----------")
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)
    logg.info(f"confusion matrix : ")
    cfm = pd.DataFrame(
        100 * confusion_matrix(y_true, y_pred, normalize="all").round(4),
        columns=le.classes_,
        index=le.classes_,
    )
    logg.info(cfm.to_string())
    f1 = 100 * f1_score(y_true, y_pred, average="macro").round(5)
    acc = 100 * accuracy_score(y_true, y_pred).round(5)
    logg.info(f"f1 score : {f1}")
    logg.info(f"accuracy score : {acc}")
    logg.info("----------- END REPORT -----------")

    return logg, f1, acc


def extract_best_lr(dic):
    pdf = pd.DataFrame(dic).T
    pdf.columns = ["f1", "acc"]
    pdf["learning_rate"] = pdf.index.str.split("_").str[1].astype(float)
    pdf["model"] = pdf.index.str.split("_").str[0]
    pdf.reset_index(drop=True, inplace=True)
    best_lr_per_model = pdf.groupby("model").apply(lambda x: x.loc[x["f1"].idxmax()])
    return best_lr_per_model


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
    path_results = f"results/{name_folder}/"
    path_data = "data/divergence_process.h5"
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
            logg.info(f"Model trained")
            top = {np.array(h.history["loss"])[0].round(5)}
            end = {np.array(h.history["loss"])[-1].round(5)}
            logg.info(f"Loss:  {top} -> {end}")
            top_val = {np.array(h.history["val_loss"])[0].round(5)}
            end_val = {np.array(h.history["val_loss"])[-1].round(5)}
            logg.info(f"Val loss:  {top_val} -> {end_val}")
            ypred = m.predict(X_test)
            print(ypred.shape, y_test.shape)
            logg, f1, acc = report_prediction(y_test, ypred, le, logg)
            dic[name_model] = [f1, acc]
            del m, h
    dump_pkl(dic, os.path.join(path_results, "dic.pkl"))

    ############################

    best_lr_per_model = extract_best_lr(dic)
    # print(best_lr_per_model)
    logg.info(f"best_lr_per_model : \n{best_lr_per_model}")

    pathreport = os.path.join(path_results, "report.txt")
    write_report(pathlog, pathreport)
