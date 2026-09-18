"""
Misura delle prestazioni del sistema come servizio.

Un classificatore si valuta con l'accuratezza; un servizio interattivo si valuta
anche con il tempo che fa attendere chi lo usa. Questo script misura:

  * il costo di inizializzazione (caricamento dei pesi e compilazione a caldo
    delle librerie di elaborazione del segnale), pagato una volta all'avvio;
  * la latenza a regime dei singoli stadi della catena;
  * il costo aggiuntivo della spiegazione Grad-CAM, che richiede una
    propagazione all'indietro e non solo in avanti;
  * il numero di richieste al secondo sostenibile in elaborazione sequenziale.

Uso:  python -m benchmark.latency
"""
import json
import statistics as st
import time

import numpy as np

from src import config
from src.inference import InferenceEngine

N_RUNS = 50


def _clip_sintetica(seed: int = 0, sr: int = 44100, durata: float = 2.5) -> np.ndarray:
    """Segnale di prova con struttura armonica e modulazione, a durata realistica."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, durata, int(sr * durata))
    y = 0.4 * np.sin(2 * np.pi * 180 * t) * (1 + 0.6 * np.sin(2 * np.pi * 4 * t))
    return (y + 0.04 * rng.standard_normal(t.size)).astype(np.float32), sr


def _riassumi(campioni):
    return {
        "mediana_ms": round(st.median(campioni), 2),
        "media_ms": round(st.mean(campioni), 2),
        "p95_ms": round(sorted(campioni)[int(0.95 * len(campioni)) - 1], 2),
        "min_ms": round(min(campioni), 2),
        "max_ms": round(max(campioni), 2),
    }


def main():
    print("Inizializzazione del motore di inferenza…")
    t0 = time.perf_counter()
    engine = InferenceEngine()
    init_ms = (time.perf_counter() - t0) * 1000

    y, sr = _clip_sintetica()

    # Prima richiesta, a freddo: paga la compilazione JIT delle librerie audio.
    t0 = time.perf_counter()
    engine.predict(y, sr, with_gradcam=True)
    freddo_ms = (time.perf_counter() - t0) * 1000

    risultati = {
        "parametri_modello": engine.n_params,
        "dimensione_checkpoint_kb": round(config.MODEL_PATH.stat().st_size / 1024, 1),
        "caricamento_modello_ms": round(init_ms, 1),
        "prima_richiesta_a_freddo_ms": round(freddo_ms, 1),
    }

    configurazioni = [
        ("spiegazione_con_propagazione_indietro", True, "backward"),
        ("spiegazione_stessa_passata_avanti", True, "fast"),
        ("senza_spiegazione", False, "fast"),
    ]

    # Riscaldamento di ogni percorso di codice prima di misurare.
    for _, gc, modo in configurazioni:
        engine.predict(y, sr, with_gradcam=gc, cam_mode=modo)

    for etichetta, gc, modo in configurazioni:
        stadi = {}
        for _ in range(N_RUNS):
            r = engine.predict(y, sr, with_gradcam=gc, cam_mode=modo)
            for k, v in r["timings"].items():
                stadi.setdefault(k, []).append(v)
        risultati[etichetta] = {k: _riassumi(v) for k, v in stadi.items()}
        mediana_totale = st.median(stadi["totale_ms"])
        risultati[etichetta]["richieste_al_secondo"] = round(1000 / mediana_totale, 1)

    lento = risultati["spiegazione_con_propagazione_indietro"]
    veloce = risultati["spiegazione_stessa_passata_avanti"]
    risultati["guadagno_ottimizzazione"] = {
        "costo_spiegazione_ridotto_di": f"{lento['spiegazione_ms']['mediana_ms'] / max(veloce['spiegazione_ms']['mediana_ms'], 1e-9):.0f}x",
        "latenza_totale_ridotta_del": f"{(1 - veloce['totale_ms']['mediana_ms'] / lento['totale_ms']['mediana_ms']) * 100:.0f}%",
        "richieste_al_secondo": f"{lento['richieste_al_secondo']} -> {veloce['richieste_al_secondo']}",
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "benchmark_latenza.json"
    out.write_text(json.dumps(risultati, indent=2, ensure_ascii=False))

    print(json.dumps(risultati, indent=2, ensure_ascii=False))
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
