"""
Interfaccia web del sistema di riconoscimento delle emozioni dal parlato.

L'applicazione è un semplice client del livello di servizio definito in
`src/inference.py`: riceve una registrazione dal browser o un file caricato
dall'utente, la consegna al motore di inferenza e presenta la risposta. Tutta la
logica — pre-elaborazione, rappresentazione tempo-frequenza, classificazione,
spiegazione, cronometraggio — vive nel livello di servizio e non qui, così che
l'interfaccia resti sostituibile senza toccare il sistema.

Avvio locale:   streamlit run streamlit_app.py
"""
import io

import numpy as np
import soundfile as sf
import streamlit as st

from src import config, viz
from src.inference import InferenceEngine

st.set_page_config(page_title="Riconoscimento delle emozioni dal parlato",
                   page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------
# Motore di inferenza
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Inizializzazione del sistema…")
def carica_motore():
    """
    Il modello viene caricato una sola volta per processo e condiviso fra tutte le
    sessioni. Una chiamata a vuoto paga subito la compilazione a caldo delle
    librerie di elaborazione del segnale, che altrimenti ricadrebbe sul primo
    utente sotto forma di alcuni secondi di attesa.
    """
    engine = InferenceEngine()
    silenzio = np.zeros(int(config.SAMPLE_RATE * config.DURATION), dtype=np.float32)
    engine.predict(silenzio, config.SAMPLE_RATE, with_gradcam=True)
    return engine


ENGINE = carica_motore()


def leggi_audio(file) -> tuple:
    """Decodifica il file audio ricevuto dal browser in (segnale, frequenza)."""
    dati, sr = sf.read(io.BytesIO(file.getvalue()), dtype="float32", always_2d=False)
    return dati, sr


# ----------------------------------------------------------------------
# Intestazione
# ----------------------------------------------------------------------
st.title("Riconoscimento delle emozioni dal parlato")
st.markdown(
    "Il sistema analizza **come** viene detta una frase, non cosa viene detto: "
    "l'emozione nel parlato è portata dalla prosodia — melodia, intensità, ritmo, "
    "dinamica spettrale — e non dal contenuto lessicale.\n\n"
    "La registrazione viene riportata a una durata fissa di tre secondi, convertita "
    "in uno spettrogramma log-Mel e classificata da una rete convoluzionale."
)

with st.sidebar:
    st.header("Ingresso")
    registrazione = st.audio_input("Registra la voce")
    st.caption("oppure")
    caricato = st.file_uploader("Carica un file audio", type=["wav", "mp3", "ogg", "flac", "m4a"])
    spiegazione = st.checkbox("Calcola la spiegazione visiva", value=True)

    st.divider()
    st.caption(
        f"Rete convoluzionale, {ENGINE.n_params:,}".replace(",", ".") + " parametri. "
        "Valutazione con separazione rigorosa dei parlanti: nessuna voce compare "
        "in più di una partizione."
    )

sorgente = registrazione or caricato

if sorgente is None:
    st.info("Registra la voce o carica un file audio dal pannello a sinistra.")
    st.stop()

# ----------------------------------------------------------------------
# Analisi
# ----------------------------------------------------------------------
try:
    segnale, sr = leggi_audio(sorgente)
except Exception as exc:
    st.error(f"Non è stato possibile leggere il file audio: {exc}")
    st.stop()

res = ENGINE.predict(segnale, sr, with_gradcam=spiegazione)
predetta = config.EMOTIONS_IT[res["predicted"]]
probabilita = {config.EMOTIONS_IT[e]: p for e, p in res["probabilities"].items()}
valenza = res["valence"]
valenza_top = max(valenza, key=valenza.get)

sin, des = st.columns([1, 1.7], gap="large")

with sin:
    st.metric("Emozione riconosciuta", predetta,
              help="Classe con probabilità più alta fra le otto")
    st.metric("Valenza", valenza_top.capitalize(),
              help="Proiezione delle otto emozioni sull'asse piacevole/spiacevole")

    st.markdown("**Distribuzione sulle otto emozioni**")
    st.pyplot(viz.probabilities_figure(probabilita, evidenzia=predetta))

    st.markdown("**Valenza**")
    st.pyplot(viz.probabilities_figure(
        {k.capitalize(): v for k, v in valenza.items()},
        evidenzia=valenza_top.capitalize()))

with des:
    st.pyplot(viz.waveform_figure(res["waveform"]))
    st.pyplot(viz.spectrogram_figure(res["mel"]))
    if spiegazione:
        st.pyplot(viz.gradcam_figure(res["mel"], res["cam"], predetta))

    t = res["timings"]
    righe = [
        "| Stadio | Tempo |",
        "|---|---:|",
        f"| Pre-elaborazione del segnale | {t['preprocessing_ms']:.1f} ms |",
        f"| Costruzione del tensore | {t['tensor_ms']:.1f} ms |",
        f"| Inferenza della rete | {t['inference_ms']:.1f} ms |",
    ]
    if spiegazione:
        righe.append(f"| Spiegazione (CAM) | {t['spiegazione_ms']:.1f} ms |")
    righe.append(f"| **Totale** | **{t['totale_ms']:.1f} ms** |")
    st.markdown("**Tempi di risposta**")
    st.markdown("\n".join(righe))

# ----------------------------------------------------------------------
with st.expander("Note sul funzionamento"):
    st.markdown(
        "**Probabilità.** Il sistema non restituisce solo l'etichetta vincente ma "
        "l'intera distribuzione: una predizione incerta ripartita fra due emozioni "
        "vicine è un'informazione, non un difetto.\n\n"
        "**Valenza.** Le otto emozioni sono proiettate sull'asse piacevole/spiacevole "
        "del modello circomplesso di Russell, sommando le probabilità di ciascun gruppo. "
        "La somma sfrutta l'intera distribuzione: una predizione incerta ripartita fra "
        "due emozioni negative diverse indica comunque valenza negativa.\n\n"
        "**Spiegazione.** Le zone in ambra sono le regioni del piano tempo-frequenza che "
        "hanno pesato di più sulla decisione. Le emozioni ad alta attivazione tendono ad "
        "attivare le bande medio-alte, quelle a bassa attivazione le bande basse.\n\n"
        "**Tempi.** Nella formulazione generale la spiegazione richiede una propagazione "
        "all'indietro nella rete e costa più della predizione stessa. Poiché questa "
        "architettura termina con un global average pooling e un unico strato lineare, la "
        "mappa si ottiene dalla medesima passata in avanti, a costo trascurabile.\n\n"
        "**Limiti.** Il sistema è addestrato su parlato recitato registrato in studio. "
        "Su voce spontanea, in ambiente rumoroso o in lingue diverse, le prestazioni "
        "sono verosimilmente inferiori. È uno strumento dimostrativo a scopo didattico "
        "e non va impiegato per valutare persone reali."
    )
