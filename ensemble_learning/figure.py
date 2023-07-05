import os
import numpy as np

from utils import open_pkl

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


def figure_divergence_weights(patthw):
    """Plot the weights of the learned combination of the divergence measures

    Parameters
    ----------
    patthw : str
        Path to the pickle file containing the weights
    """

    weights = open_pkl(patthw)
    w = []
    b = []
    for i in weights:
        w.append(i[0])
        b.append(i[1])
    w = np.array(w)
    b = np.array(b)
    b = b[:, np.newaxis, :]
    w = np.concatenate((w, b), axis=1)
    f, a = plt.subplots(1, 3, figsize=(15, 2.8), sharey=True)
    titles = ["Forêt", "Prairie", "Divergent"]
    names_x = [
        "K GAUSS COP",
        "K EFGM COP",
        "L EFGM COP",
        "L CV HH",
        "L CV HV",
        "L CV R",
        "K WBUL HH",
        "K WBUL HV",
        "K WBUL R",
        "L MU HH",
        "L MU HV",
        "L MU R",
        "K RGGA HH",
        "K RGGA HV",
        "K RGGA R",
        "L VAR HH",
        "L VAR HV",
        "L VAR R",
        "BIAIS",
    ]
    col = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(3):
        a[i].bar(
            names_x,
            np.mean(w[:, :, i], axis=0),
            color=col[i],
            alpha=0.6,
            label="mean",
            yerr=[
                np.mean(w[:, :, i], axis=0) - np.min(w[:, :, i], axis=0),
                np.max(w[:, :, i], axis=0) - np.mean(w[:, :, i], axis=0),
            ],
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        a[i].set_yscale("log")
        a[i].xaxis.set_tick_params(rotation=90)
        a[i].set_title(titles[i], fontsize=15, fontweight="bold")
        if i == 0:
            a[i].set_ylabel("Valeur Poids", fontsize=15, fontweight="bold")
    plt.tight_layout()
    drname = os.path.dirname(patthw).split("/")[-1]
    plt.savefig(
        os.path.join(f"../figures/{drname}_weights.pdf"),
        bbox_inches="tight",
        backend="pgf",
    )


if __name__ == "__main__":
    patthw = "../results/300623_15H32M00/weights_arch1.pkl"
    figure_divergence_weights(patthw)
