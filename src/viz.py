"""
Figure per l'interfaccia web.

Due immagini, con ruoli distinti e quindi codifiche visive distinte:

  * lo spettrogramma log-Mel mostra il *segnale*: una grandezza scalare
    (energia in decibel) su un piano tempo-frequenza, quindi una scala
    sequenziale percettivamente uniforme;

  * la mappa Grad-CAM mostra la *spiegazione*, ed è sovrapposta al segnale reso
    in scala di grigi. Tenere il segnale acromatico e riservare il colore alla
    spiegazione evita che due mappe cromatiche competano nella stessa immagine:
    si vede a colpo d'occhio quale strato è il dato e quale l'interpretazione.
"""
import matplotlib
matplotlib.use("Agg")                      # backend senza finestra grafica
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src import config

# Rampa sequenziale a tinta unica per la mappa di salienza: dal trasparente
# all'ambra satura. L'opacità cresce con il valore, così le regioni irrilevanti
# lasciano vedere lo spettrogramma sottostante invece di coprirlo.
_CAM_CMAP = LinearSegmentedColormap.from_list(
    "salienza",
    [(1.0, 0.85, 0.40, 0.00), (0.98, 0.65, 0.16, 0.55), (0.85, 0.28, 0.06, 0.90)],
)

# Versione opaca della stessa rampa, usata solo per la barra di riferimento:
# una legenda semitrasparente su fondo bianco risulterebbe slavata.
_CAM_CMAP_SOLID = LinearSegmentedColormap.from_list(
    "salienza_piena",
    [(1.0, 0.95, 0.80), (0.98, 0.65, 0.16), (0.85, 0.28, 0.06)],
)

_TIME_MAX = config.DURATION
_FIGSIZE = (7.2, 3.1)
_DPI = 110


def _style_axes(ax):
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_ylabel("Banda Mel", fontsize=9)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")


def spectrogram_figure(mel: np.ndarray):
    """Spettrogramma log-Mel in decibel."""
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, _TIME_MAX, 0, mel.shape[0]])
    ax.set_title("Spettrogramma log-Mel", fontsize=10, pad=6)
    _style_axes(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("dB", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    return fig


def gradcam_figure(mel: np.ndarray, cam: np.ndarray, emotion_it: str):
    """Spettrogramma in scala di grigi con la mappa di salienza sovrapposta."""
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="Greys_r",
              extent=[0, _TIME_MAX, 0, mel.shape[0]])
    ax.imshow(cam, origin="lower", aspect="auto", cmap=_CAM_CMAP,
              vmin=0, vmax=1, extent=[0, _TIME_MAX, 0, mel.shape[0]])
    ax.set_title(f"Regioni che determinano la predizione «{emotion_it}»",
                 fontsize=10, pad=6)
    _style_axes(ax)

    ref = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1), cmap=_CAM_CMAP_SOLID)
    cb = fig.colorbar(ref, ax=ax, pad=0.015)
    cb.set_label("salienza", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    return fig


def waveform_figure(wav: np.ndarray):
    """Forma d'onda dopo la normalizzazione a lunghezza fissa."""
    t = np.linspace(0, _TIME_MAX, len(wav))
    fig, ax = plt.subplots(figsize=(_FIGSIZE[0], 1.6), dpi=_DPI)
    ax.plot(t, wav, linewidth=0.6, color="#3b6fb6")
    ax.set_xlim(0, _TIME_MAX)
    ax.set_title("Forma d'onda normalizzata (3 s)", fontsize=10, pad=6)
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_yticks([])
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#9aa0a6")
    fig.tight_layout()
    return fig
