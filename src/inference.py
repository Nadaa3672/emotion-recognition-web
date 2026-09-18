"""
Livello di servizio: trasforma il modello addestrato in un componente
interrogabile da un'applicazione web.

Rispetto a uno script di valutazione offline cambiano due cose. La prima è che
l'ingresso non è una clip di un corpus ma una registrazione arbitraria
dell'utente, di durata e qualità ignote: la catena deve essere robusta.
La seconda è che conta il *tempo di risposta*: un servizio interattivo viene
giudicato anche da quanto attende chi lo usa, quindi ogni stadio viene
cronometrato separatamente.

Il modello è caricato una volta sola e tenuto in memoria (`get_engine`): il
caricamento dei pesi costa più dell'inferenza stessa e ripeterlo a ogni
richiesta sarebbe l'errore di progettazione più comune in questo scenario.
"""
import time

import numpy as np
import torch

from src import config, preprocessing, model as model_mod, gradcam as gradcam_mod


class InferenceEngine:
    """Modello, statistiche di normalizzazione ed estrattore Grad-CAM, caricati una volta sola."""

    def __init__(self, model_path=None, device="cpu"):
        t0 = time.perf_counter()
        self.device = torch.device(device)
        self.model, self.mean, self.std = model_mod.load_model(model_path, self.device)
        blocco = gradcam_mod.last_conv_block(self.model)
        # Entrambe le implementazioni condividono lo stesso hook sulle attivazioni.
        self.cam_fast = gradcam_mod.FastCAM(self.model, blocco,
                                            gradcam_mod.final_linear(self.model))
        self.cam_grad = gradcam_mod.GradCAM(self.model, blocco)
        self.load_time_ms = (time.perf_counter() - t0) * 1000
        self.n_params = model_mod.n_parameters(self.model)

    # ------------------------------------------------------------------
    def predict(self, y_raw: np.ndarray, sr_in: int, with_gradcam: bool = True,
                cam_mode: str = "fast") -> dict:
        """
        Catena completa su una registrazione.

        `cam_mode` seleziona l'implementazione della spiegazione: "fast" riusa la
        passata in avanti già eseguita per la predizione, "backward" ricorre alla
        formulazione generale con propagazione all'indietro. Le due producono la
        stessa mappa; la seconda serve come riferimento nel confronto dei tempi.
        """
        timings = {}

        t0 = time.perf_counter()
        wav, mel = preprocessing.prepare(y_raw, sr_in)
        timings["preprocessing_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        x = model_mod.to_tensor(mel, self.mean, self.std).to(self.device)
        timings["tensor_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        timings["inference_ms"] = (time.perf_counter() - t0) * 1000

        cam = None
        if with_gradcam:
            t0 = time.perf_counter()
            if cam_mode == "fast":
                cam, _ = self.cam_fast(x, logits=logits)
            else:
                cam, _ = self.cam_grad(x)
            timings["spiegazione_ms"] = (time.perf_counter() - t0) * 1000

        timings["totale_ms"] = sum(timings.values())

        probabilities = {e: float(p) for e, p in zip(config.EMOTIONS, probs)}
        return {
            "probabilities": probabilities,
            "predicted": config.EMOTIONS[int(np.argmax(probs))],
            "valence": valence_scores(probabilities),
            "waveform": wav,
            "mel": mel,
            "cam": cam,
            "timings": timings,
        }


def valence_scores(probabilities: dict) -> dict:
    """
    Proietta la distribuzione sulle otto emozioni sui tre livelli di valenza,
    sommando le probabilità delle emozioni che appartengono a ciascun gruppo.

    Si somma sulle probabilità e non sull'etichetta vincente perché una
    predizione incerta distribuita su due emozioni negative diverse indica
    comunque, con buona confidenza, una valenza negativa.
    """
    out = {v: 0.0 for v in config.VALENCE_CLASSES}
    for emotion, p in probabilities.items():
        out[config.VALENCE_MAP[emotion]] += float(p)
    return out


# ----------------------------------------------------------------------
# Istanza condivisa dall'applicazione
# ----------------------------------------------------------------------
_ENGINE = None


def get_engine(model_path=None) -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine(model_path)
    return _ENGINE
