"""
Figure della valutazione sperimentale.

Tre immagini, ciascuna con una codifica visiva scelta in base al lavoro che deve
fare: matrici di confusione come mappe di calore a tinta unica (una grandezza
scalare, quindi scala sequenziale chiaro-scuro); F1 per classe come barre
orizzontali ordinate (confronto di magnitudine fra categorie nominali).

Uso:  python -m evaluation.figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config

INK = "#1f2328"
INK_SOFT = "#5a6169"
GRID = "#d7dbe0"
BAR = "#3b6fb6"
BAR_WEAK = "#a8c0dd"

EMO_IT = [config.EMOTIONS_IT[e] for e in config.EMOTIONS]


def _pulisci(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=INK_SOFT)


def matrice_confusione(cm, etichette, titolo, path, normalizza=True):
    """Mappa di calore a tinta unica: il valore è una quantità, non un'identità."""
    cm = np.asarray(cm, dtype=float)
    quota = cm / cm.sum(axis=1, keepdims=True)
    dati = quota if normalizza else cm

    n = len(etichette)
    fig, ax = plt.subplots(figsize=(0.78 * n + 2.6, 0.72 * n + 2.2), dpi=130)
    im = ax.imshow(dati, cmap="Blues", vmin=0, vmax=dati.max())

    ax.set_xticks(range(n), etichette, rotation=45, ha="right")
    ax.set_yticks(range(n), etichette)
    ax.set_xlabel("Predetta", fontsize=10, color=INK)
    ax.set_ylabel("Reale", fontsize=10, color=INK)
    ax.set_title(titolo, fontsize=11, color=INK, pad=10)
    _pulisci(ax)

    soglia = dati.max() * 0.55
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            testo = f"{int(cm[i, j])}" if not normalizza else f"{quota[i, j]:.0%}"
            ax.text(j, i, testo, ha="center", va="center", fontsize=8.5,
                    color="white" if dati[i, j] > soglia else INK)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("quota della classe reale" if normalizza else "clip", fontsize=8, color=INK_SOFT)
    cb.ax.tick_params(labelsize=7, length=0, colors=INK_SOFT)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def f1_per_classe(report, path):
    """Barre orizzontali ordinate: una sola serie, quindi nessuna legenda."""
    valori = [(config.EMOTIONS_IT[e], report[e]["f1-score"]) for e in config.EMOTIONS]
    valori.sort(key=lambda kv: kv[1])
    nomi = [v[0] for v in valori]
    f1 = [v[1] for v in valori]
    mediana = float(np.median(f1))

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=130)
    colori = [BAR if v >= mediana else BAR_WEAK for v in f1]
    barre = ax.barh(nomi, f1, color=colori, height=0.62)

    for b, v in zip(barre, f1):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=8.5, color=INK)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1", fontsize=10, color=INK)
    ax.set_title("Prestazione per emozione", fontsize=11, color=INK, pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _pulisci(ax)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def confronto_riferimenti(res, path):
    """Accuratezza del sistema contro i riferimenti, nei due compiti."""
    otto = res["otto_classi"]
    val = res["valenza"]

    gruppi = [
        ("Otto emozioni", [
            ("Caso uniforme", otto["caso_uniforme"]),
            ("Classe maggioritaria", otto["riferimento_classe_maggioritaria"]["accuratezza"]),
            ("SVM su MFCC", otto["SVM_su_MFCC"]["accuratezza"]),
            ("Rete convoluzionale", otto["CNN"]["accuratezza"]),
        ]),
        ("Valenza (tre classi)", [
            ("Classe maggioritaria", val["riferimento_classe_maggioritaria"]["accuratezza"]),
            ("SVM su MFCC", val["SVM_su_MFCC"]["accuratezza"]),
            ("Rete convoluzionale", val["A_decisione_poi_proiezione"]["accuratezza"]),
        ]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=130,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, (titolo, voci) in zip(axes, gruppi):
        nomi = [v[0] for v in voci]
        vals = [v[1] for v in voci]
        colori = [BAR_WEAK] * (len(vals) - 1) + [BAR]
        barre = ax.barh(nomi, vals, color=colori, height=0.6)
        for b, v in zip(barre, vals):
            ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
        ax.set_xlabel("Accuratezza", fontsize=9.5, color=INK)
        ax.set_title(titolo, fontsize=11, color=INK, pad=10)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        _pulisci(ax)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    res = json.loads((config.RESULTS_DIR / "valutazione.json").read_text())
    fig_dir = config.RESULTS_DIR / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)

    matrice_confusione(res["otto_classi"]["matrice_confusione"], EMO_IT,
                       "Matrice di confusione — otto emozioni",
                       fig_dir / "confusione_8.png")
    matrice_confusione(res["valenza"]["matrice_confusione_B"],
                       [v.capitalize() for v in config.VALENCE_CLASSES],
                       "Matrice di confusione — valenza",
                       fig_dir / "confusione_valenza.png")
    f1_per_classe(res["otto_classi"]["report_per_classe"], fig_dir / "f1_per_classe.png")
    confronto_riferimenti(res, fig_dir / "confronto_riferimenti.png")

    print(f"Figure salvate in {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
