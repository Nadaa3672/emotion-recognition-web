"""
Interfaccia web del sistema di riconoscimento delle emozioni dal parlato.

L'applicazione è un client del livello di servizio definito in `src/inference.py`:
riceve una registrazione dal browser o un file caricato dall'utente, la consegna
al motore di inferenza e presenta la risposta. Tutta la logica — pre-elaborazione,
rappresentazione tempo-frequenza, classificazione, spiegazione, cronometraggio —
vive nel livello di servizio e non qui.

Due accorgimenti riguardano il comportamento dell'applicazione sotto carico.
Il modello è caricato una volta per processo e condiviso fra le sessioni. L'analisi
è memorizzata in cache sul contenuto della registrazione: Streamlit riesegue lo
script a ogni interazione, e senza cache ogni click ricalcolerebbe l'intera catena,
bloccando il processo mentre il componente di registrazione attende risposta.

Avvio locale:   streamlit run streamlit_app.py
"""
import hashlib
import io

import numpy as np
import soundfile as sf
import streamlit as st

from src import config, viz
from src.inference import InferenceEngine

st.set_page_config(page_title="Riconoscimento delle emozioni dal parlato",
                   page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------
# Motore di inferenza (una istanza per processo)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Inizializzazione del sistema…")
def carica_motore():
    engine = InferenceEngine()
    silenzio = np.zeros(int(config.SAMPLE_RATE * config.DURATION), dtype=np.float32)
    engine.predict(silenzio, config.SAMPLE_RATE, with_gradcam=True)   # riscaldamento
    return engine


ENGINE = carica_motore()


@st.cache_data(show_spinner=False, max_entries=8)
def analizza(audio_bytes: bytes, spiegazione: bool, _chiave: str):
    """
    Esegue la catena su una registrazione e restituisce solo dati serializzabili.

    La chiave di cache è l'impronta del contenuto audio: ripetere l'analisi della
    stessa registrazione non ricalcola nulla.
    """
    segnale, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    res = ENGINE.predict(segnale, sr, with_gradcam=spiegazione)
    return {
        "predicted": res["predicted"],
        "probabilities": res["probabilities"],
        "valence": res["valence"],
        "waveform": res["waveform"],
        "mel": res["mel"],
        "cam": res["cam"],
        "timings": res["timings"],
    }


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
    st.header("Opzioni")
    spiegazione = st.checkbox("Calcola la spiegazione visiva (Grad-CAM)", value=True)
   
    st.divider()
    st.caption(
        f"Rete convoluzionale, {ENGINE.n_params:,}".replace(",", ".") + " parametri. "
        "Valutazione con separazione rigorosa dei parlanti: nessuna voce compare "
        "in più di una partizione."
    )

# ----------------------------------------------------------------------
# Ingresso
# ----------------------------------------------------------------------
st.subheader("Ingresso")
col_mic, col_file = st.columns(2)
with col_mic:
    registrazione = st.audio_input("Registra la voce")
with col_file:
    caricato = st.file_uploader("oppure carica un file audio",
                                type=["wav", "mp3", "ogg", "flac", "m4a"])

sorgente = registrazione or caricato

if sorgente is None:
    st.info("Registra la voce o carica un file audio per avviare l'analisi.")
    st.stop()

audio_bytes = sorgente.getvalue()
impronta = hashlib.sha1(audio_bytes).hexdigest()

try:
    res = analizza(audio_bytes, spiegazione, impronta)
except Exception as exc:
    st.error(f"Non è stato possibile elaborare l'audio: {exc}")
    st.stop()

predetta = config.EMOTIONS_IT[res["predicted"]]
probabilita = {config.EMOTIONS_IT[e]: p for e, p in res["probabilities"].items()}
valenza = res["valence"]
valenza_top = max(valenza, key=valenza.get)

# ----------------------------------------------------------------------
# Risultati
# ----------------------------------------------------------------------
st.divider()
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
    if spiegazione and res["cam"] is not None:
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
        righe.append(f"| Spiegazione Grad-CAM | {t['spiegazione_ms']:.1f} ms |")
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
        "**Grad-CAM.** Le zone calde sono le regioni del piano tempo-frequenza che hanno "
        "pesato di più sulla decisione. Le emozioni ad alta attivazione tendono ad "
        "attivare le bande medio-alte, quelle a bassa attivazione le bande basse.\n\n"
        "**Tempi.** Nella formulazione generale Grad-CAM richiede una propagazione "
        "all'indietro nella rete e costa più della predizione stessa. Poiché questa "
        "architettura termina con un global average pooling e un unico strato lineare, "
        "la mappa — identica a meno dell'errore di arrotondamento — si ottiene dalla "
        "medesima passata in avanti, a costo trascurabile.\n\n"
        "**Limiti.** Il sistema è addestrato su parlato recitato registrato in studio. "
        "Su voce spontanea, in ambiente rumoroso o in lingue diverse, le prestazioni "
        "sono verosimilmente inferiori. È uno strumento dimostrativo a scopo didattico "
        "e non va impiegato per valutare persone reali."
    )
