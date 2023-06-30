from explore_learning import *
from sklearn.model_selection import StratifiedKFold


def Kfold_learning_rate(
    model, learning_rate, X, y, frac_val, batch_size, n_epochs, callbacks, logg
):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    kfold = 0
    f1 = []
    acc = []
    le = LabelEncoder()
    y = le.fit_transform(y)
    weights = []
    for train_index, test_index in skf.split(X, y):
        logg.info(f"Kfold : {kfold}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=frac_val
        )
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)
        m = model(learning_rate, num_classes=y_train.shape[1])
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
        top = {h.history["loss"][0]}
        end = {h.history["loss"][-1]}
        logg.info(f"Loss:  {top} -> {end}")
        top_val = {h.history["val_loss"][0]}
        end_val = {h.history["val_loss"][-1]}
        logg.info(f"Val loss:  {top_val} -> {end_val}")
        logg.info(f"Model trained")
        ypred = m.predict(X_test)
        logg, f1_k, acc_k = report_prediction(y_test, ypred, le, logg)
        f1.append(f1_k)
        acc.append(acc_k)
        kfold += 1
        weights.append(m.get_weights())
        del m
    return f1, acc, logg, weights


def report_kfold(dic, logg):
    logg.info(f"======== Final report ========")
    for i in list(dic.keys()):
        logg.info(f"-------- Model : {i} --------")
        f1 = dic[i]["f1"]
        acc = dic[i]["acc"]
        logg.info(f"f1 : {np.mean(f1)} +/- {np.std(f1)}")
        logg.info(f"acc : {np.mean(acc)} +/- {np.std(acc)}")
    logg.info(f"======== End report ========")
    return logg


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

    f1a, acca, logg, weighta = Kfold_learning_rate(
        arch1, 0.15, X, y, frac_val, batch_size, n_epochs, callbacks, logg
    )
    dump_pkl(weighta, os.path.join(path_results, "weights_arch1.pkl"))

    f1b, accb, logg, weightb = Kfold_learning_rate(
        arch2, 0.0375, X, y, frac_val, batch_size, n_epochs, callbacks, logg
    )
    dump_pkl(weightb, os.path.join(path_results, "weights_arch2.pkl"))

    f1c, accc, logg, weightc = Kfold_learning_rate(
        arch4, 0.0375, X, y, frac_val, batch_size, n_epochs, callbacks, logg
    )
    dump_pkl(weightc, os.path.join(path_results, "weights_arch4.pkl"))

    dic = {
        "arch1": {"f1": f1a, "acc": acca},
        "arch2": {"f1": f1b, "acc": accb},
        "arch4": {"f1": f1c, "acc": accc},
    }
    dump_pkl(dic, os.path.join(path_results, "kfold_dic.pkl"))

    logg = report_kfold(dic, logg)
    pathreport = os.path.join(path_results, "report.txt")
    write_report(pathlog, pathreport)
