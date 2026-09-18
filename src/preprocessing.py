"""
Pre-elaborazione del segnale audio.

Ogni registrazione, qualunque sia la sua origine (microfono del browser, file
caricato dall'utente, clip di un corpus), viene ricondotta a una forma d'onda di
lunghezza fissa, così che la rappresentazione tempo-frequenza calcolata a valle
abbia sempre la stessa dimensione.

Catena:  caricamento -> mono -> ricampionamento -> rimozione del silenzio
         -> lunghezza fissa -> spettrogramma log-Mel
"""
import numpy as np
import librosa

from src import config


def load_audio(path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Carica un file audio come segnale mono in virgola mobile alla frequenza target."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def from_array(y: np.ndarray, sr_in: int, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Normalizza un segnale già in memoria (il caso del microfono del browser, che
    consegna un array di interi alla frequenza della scheda audio).
    """
    y = np.asarray(y)
    if y.ndim > 1:                       # stereo -> mono
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):   # interi -> float in [-1, 1]
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    y = y.astype(np.float32)
    if sr_in != sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = config.TRIM_TOP_DB) -> np.ndarray:
    """
    Rimuove il silenzio iniziale e finale.

    Senza questo passo una registrazione con molto silenzio davanti sposterebbe
    il contenuto utile fuori dalla finestra di analisi.
    """
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y


def fix_length(y: np.ndarray, sr: int = config.SAMPLE_RATE,
               duration: float = config.DURATION) -> np.ndarray:
    """Porta il segnale alla durata fissa: padding simmetrico se corto, ritaglio centrale se lungo."""
    target = int(sr * duration)
    if len(y) < target:
        pad = target - len(y)
        left = pad // 2
        y = np.pad(y, (left, pad - left), mode="constant")
    elif len(y) > target:
        start = (len(y) - target) // 2
        y = y[start:start + target]
    return y.astype(np.float32)


def log_mel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Spettrogramma log-Mel in decibel, di forma (N_MELS, T).

    La scala Mel comprime le alte frequenze riproducendo la risoluzione non
    uniforme dell'udito umano; la conversione in decibel rende la dinamica
    logaritmica, come la percezione dell'intensità sonora.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def prepare(y: np.ndarray, sr_in: int) -> tuple:
    """
    Catena completa a partire da un segnale grezzo.
    Restituisce (forma d'onda a lunghezza fissa, spettrogramma log-Mel).
    """
    y = from_array(y, sr_in)
    y = trim_silence(y)
    y = fix_length(y)
    return y, log_mel(y)
