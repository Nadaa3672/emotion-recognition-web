"""
Configurazione globale del sistema.

Raccoglie in un unico punto i parametri audio, le etichette delle emozioni e i
percorsi, in modo che l'applicazione web e gli script di valutazione usino
esattamente le stesse impostazioni del modello addestrato.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Percorsi
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_cnn.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

# ----------------------------------------------------------------------
# Parametri audio e di estrazione delle feature
# Devono coincidere con quelli usati in addestramento: cambiarli invalida il
# modello, perché la rete si aspetta un input di forma (3, 128, 188).
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000     # frequenza di campionamento
DURATION = 3.0          # durata fissa in secondi (padding o ritaglio centrale)
N_MELS = 128            # numero di bande Mel
N_FFT = 1024            # dimensione della finestra FFT
HOP_LENGTH = 256        # passo tra finestre successive
TRIM_TOP_DB = 25        # soglia per la rimozione del silenzio

# ----------------------------------------------------------------------
# Etichette
# L'ordine è vincolante: corrisponde agli indici di uscita della rete.
# ----------------------------------------------------------------------
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Traduzioni per l'interfaccia
EMOTIONS_IT = {
    "neutral": "Neutrale",
    "calm": "Calmo",
    "happy": "Felice",
    "sad": "Triste",
    "angry": "Arrabbiato",
    "fearful": "Impaurito",
    "disgust": "Disgustato",
    "surprised": "Sorpreso",
}

# ----------------------------------------------------------------------
# Proiezione sull'asse della valenza (modello circomplesso di Russell).
# Serve a offrire all'utente una sintesi a grana grossa accanto alla
# predizione dettagliata sulle otto emozioni.
# ----------------------------------------------------------------------
VALENCE_MAP = {
    "happy": "positiva",
    "surprised": "positiva",
    "angry": "negativa",
    "sad": "negativa",
    "fearful": "negativa",
    "disgust": "negativa",
    "neutral": "neutra",
    "calm": "neutra",
}
VALENCE_CLASSES = ["positiva", "neutra", "negativa"]
