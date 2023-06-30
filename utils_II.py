import re, os
import logging, pickle
from datetime import datetime
import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1


def open_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def open_log_file(path_log):
    with open(path_log, "r") as f:
        log = f.readlines()
    return log


def clean_log(log):
    result = []
    for i in range(len(log)):
        pattern = r"Line: \d+ - (.*)$"
        match = re.search(pattern, log[i])

        if match:
            result.append(match.group(1) + "\n")
        else:
            result.append(log[i])
    return result


def write_report(path_log, path_report):
    op = open_log_file(path_log)
    result = clean_log(op)
    with open(path_report, "w") as f:
        f.writelines(result)
    return result


def init_logger(path_log):
    now = datetime.now()
    namlog = now.strftime("%d%m%y_%HH%MM%S")
    datestr = "%m/%d/%Y-%I:%M:%S %p "
    filename = os.path.join(path_log, f"log_{namlog}.log")
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        format="%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s",
    )
    logging.info("Started")
    return logging, filename
